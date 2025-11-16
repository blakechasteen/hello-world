#!/usr/bin/env python3
"""
Progress Bars Showcase Demo
============================

Demonstrates the Rich progress bars added to HoloLoom demo scripts.
Shows different progress bar patterns and styles.

Usage:
    PYTHONPATH=. python demos/demo_progress_bars_showcase.py

Created: 2025-11-16
"""

import asyncio
import random
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

console = Console()


def demo_simple_progress():
    """Demonstrate simple counted loop progress."""
    console.print(Panel.fit(
        "[bold cyan]Demo 1: Simple Counted Loop[/bold cyan]",
        border_style="cyan"
    ))

    items = ["Query Thompson Sampling", "Explain Exploration-Exploitation",
             "Compare PPO and DDPG", "Describe HoloLoom", "Review Best Practices"]

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
        task = progress.add_task("[cyan]Processing queries[/cyan]", total=len(items))

        for i, item in enumerate(items, 1):
            progress.update(
                task,
                description=f"[cyan]Query {i}/{len(items)}: {item[:30]}...[/cyan]"
            )
            time.sleep(0.5)  # Simulate work
            progress.advance(task)

    console.print("[green]✓ Demo 1 complete![/green]\n")


def demo_nested_progress():
    """Demonstrate multiple sequential progress bars."""
    console.print(Panel.fit(
        "[bold cyan]Demo 2: Sequential Progress Bars[/bold cyan]",
        border_style="cyan"
    ))

    # First phase
    console.print("[bold]Phase 1: Ingesting Documents[/bold]")
    documents = ["article1.pdf", "article2.pdf", "article3.pdf", "article4.pdf"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Ingesting[/cyan]", total=len(documents))

        for doc in documents:
            progress.update(task, description=f"[cyan]Reading: {doc}[/cyan]")
            time.sleep(0.3)
            progress.advance(task)

    console.print("[green]✓ Phase 1 complete!\n[/green]")

    # Second phase
    console.print("[bold]Phase 2: Processing Embeddings[/bold]")
    chunks = list(range(1, 7))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Processing[/cyan]", total=len(chunks))

        for chunk in chunks:
            progress.update(task, description=f"[cyan]Chunk {chunk}/6[/cyan]")
            time.sleep(0.2)
            progress.advance(task)

    console.print("[green]✓ Phase 2 complete!\n[/green]")


def demo_indeterminate_progress():
    """Demonstrate indeterminate progress (spinner only)."""
    console.print(Panel.fit(
        "[bold cyan]Demo 3: Indeterminate Progress (Spinner)[/bold cyan]",
        border_style="cyan"
    ))

    operations = [
        "Building knowledge graph...",
        "Computing embeddings...",
        "Extracting entities...",
        "Caching results...",
        "Finalizing..."
    ]

    for op in operations:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"[cyan]{op}[/cyan]")
            time.sleep(0.4)
            progress.update(task, completed=100)

    console.print("[green]✓ Demo 3 complete!\n[/green]")


def demo_realistic_dashboard():
    """Demonstrate realistic dashboard demo progress."""
    console.print(Panel.fit(
        "[bold cyan]Demo 4: Realistic Dashboard Demo[/bold cyan]\\n"
        "[dim]Simulates demo_dashboard.py progress bars[/dim]",
        border_style="cyan"
    ))

    # Queries phase
    console.print("[bold]Phase 1: Executing Sample Queries[/bold]")
    queries = [
        "What is Thompson Sampling?",
        "Explain exploration-exploitation tradeoff",
        "How does PPO differ from DDPG?",
        "What are HoloLoom features?",
        "Is recursive reasoning beneficial?",
        "How to architect scalable API?"
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
        task = progress.add_task("[cyan]Executing queries[/cyan]", total=len(queries))

        for i, query in enumerate(queries, 1):
            progress.update(
                task,
                description=f"[cyan]Query {i}/{len(queries)}: {query[:35]}...[/cyan]"
            )
            time.sleep(0.4)  # Simulate weaving
            progress.advance(task)

    console.print("[green]✓ All queries executed!\n[/green]")

    # Skills phase
    console.print("[bold]Phase 2: Executing Sample Skills[/bold]")
    skills = ["code-reviewer", "test-generator", "refactoring-expert"]

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
        task = progress.add_task("[cyan]Executing skills[/cyan]", total=len(skills))

        for i, skill in enumerate(skills, 1):
            progress.update(
                task,
                description=f"[cyan]Skill {i}/{len(skills)}: {skill}[/cyan]"
            )
            time.sleep(0.3)
            progress.advance(task)

    console.print("[green]✓ All skills executed!\n[/green]")


def demo_rag_generation():
    """Demonstrate RAG query generation progress."""
    console.print(Panel.fit(
        "[bold cyan]Demo 5: RAG Query Generation[/bold cyan]\\n"
        "[dim]Simulates demo_rag_dashboard.py query generation[/dim]",
        border_style="cyan"
    ))

    topics = [
        "Thompson Sampling",
        "Bayesian inference",
        "Reinforcement learning",
        "Multi-armed bandits",
        "Exploration-exploitation",
        "Neural networks",
        "Transformers",
        "Attention mechanisms",
        "BERT embeddings",
        "RAG systems",
        "Knowledge graphs",
        "Semantic search",
        "Embeddings",
        "Vector databases",
        "LLM integration"
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
            "[cyan]Generating sample queries[/cyan]",
            total=len(topics)
        )

        for i, topic in enumerate(topics, 1):
            progress.update(
                task,
                description=f"[cyan]Query {i}/{len(topics)}: {topic}[/cyan]"
            )
            time.sleep(0.15)  # Fast generation
            progress.advance(task)

    console.print("[green]✓ Queries generated!\n[/green]")


async def main():
    """Run all demos."""
    console.print(Panel.fit(
        "[bold white on blue] Progress Bars Showcase [/bold white on blue]\\n\\n"
        "[italic]Demonstrating Rich progress bars in HoloLoom demos[/italic]",
        border_style="blue"
    ))

    console.print("[dim]This demo shows 5 different progress bar patterns\\n[/dim]")

    try:
        demo_simple_progress()
        demo_nested_progress()
        demo_indeterminate_progress()
        demo_realistic_dashboard()
        demo_rag_generation()

        console.print("\\n" + "=" * 70)
        console.print("[green bold]✅ All demos complete![/green bold]")
        console.print("=" * 70)

        console.print("\\n[bold cyan]Progress Bar Patterns Used:[/bold cyan]")
        console.print("  1. [cyan]Simple Counted Loop[/cyan] - Queries with counter")
        console.print("  2. [cyan]Sequential Progress[/cyan] - Multiple phases with bars")
        console.print("  3. [cyan]Indeterminate Progress[/cyan] - Spinner-only bars")
        console.print("  4. [cyan]Realistic Dashboard[/cyan] - Demo dashboard pattern")
        console.print("  5. [cyan]RAG Generation[/cyan] - Query generation pattern")

        console.print("\\n[bold cyan]Key Features:[/bold cyan]")
        console.print("  • [green]Spinner animation[/green] - Visual feedback")
        console.print("  • [green]Progress bar[/green] - Percentage filled")
        console.print("  • [green]Counter[/green] - X/Y display")
        console.print("  • [green]Elapsed time[/green] - Time spent")
        console.print("  • [green]Time remaining[/green] - ETA")
        console.print("  • [green]Transient[/green] - Disappears after completion")

        console.print("\\n[bold cyan]Implementation Files:[/bold cyan]")
        console.print("  • [cyan]demos/demo_dashboard.py[/cyan] - Dashboard demo with progress")
        console.print("  • [cyan]demos/demo_rag_dashboard.py[/cyan] - RAG demo with progress")
        console.print("  • [cyan]PROGRESS_BARS_IMPLEMENTATION.md[/cyan] - Complete guide")
        console.print("  • [cyan]PROGRESS_BARS_QUICK_REFERENCE.md[/cyan] - Quick ref")

    except KeyboardInterrupt:
        console.print("\\n[yellow]Demo interrupted by user[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
