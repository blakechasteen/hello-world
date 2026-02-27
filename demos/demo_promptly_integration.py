#!/usr/bin/env python3
"""
Promptly → HoloLoom Integration Demo
====================================

Demonstrates the elegant integration of Promptly's recursive intelligence
into HoloLoom's weaving architecture.

Shows:
1. Direct weaving (no refinement)
2. Automatic refinement (quality-driven)
3. Explicit strategies (REFINE, CRITIQUE, DECOMPOSE, EXPLORE, HOFSTADTER)
4. Reasoning provenance via ReasoningJournal
5. Comparison of strategies

Usage:
    PYTHONPATH=. python demos/demo_promptly_integration.py
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown

from hololoom.config import Config
from hololoom.documentation.types import Query, MemoryShard
from hololoom.weaving_orchestrator_recursive import RecursiveWeavingOrchestrator
from hololoom.protocols.recursive_reasoning import ReasoningStrategy


console = Console()


def create_test_shards():
    """Create sample memory shards for testing"""
    return [
        MemoryShard(
            content="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            entities=["Thompson Sampling", "Bayesian", "multi-armed bandit"],
            motifs=["exploration", "exploitation"]
        ),
        MemoryShard(
            content="Recursive loops enable iterative refinement of responses.",
            entities=["recursive loops", "refinement"],
            motifs=["iteration", "improvement"]
        ),
        MemoryShard(
            content="Hofstadter's strange loops involve self-reference and meta-cognition.",
            entities=["Hofstadter", "strange loops", "meta-cognition"],
            motifs=["recursion", "self-reference"]
        )
    ]


async def demo_1_basic_weaving():
    """Demo 1: Basic weaving without refinement"""
    console.print(Panel.fit(
        "[bold cyan]Demo 1: Basic Weaving (No Refinement)[/bold cyan]\n\n"
        "Standard HoloLoom weaving with recursive reasoning disabled.",
        border_style="cyan"
    ))

    config = Config.fast()
    shards = create_test_shards()

    # Create orchestrator with recursion disabled
    orchestrator = RecursiveWeavingOrchestrator(
        cfg=config,
        shards=shards,
        enable_recursive=False
    )

    query = Query(text="What is Thompson Sampling?")

    spacetime = await orchestrator.weave(query)

    console.print(f"\n[bold]Query:[/bold] {query.text}")
    console.print(f"[bold]Response:[/bold] {spacetime.response}")
    console.print(f"[bold]Confidence:[/bold] {spacetime.confidence:.2f}")
    console.print(f"[bold]Iterations:[/bold] {spacetime.iterations}")
    console.print(f"[bold]Strategy:[/bold] {spacetime.strategy_used.value if spacetime.strategy_used else 'None'}")


async def demo_2_automatic_refinement():
    """Demo 2: Automatic quality-driven refinement"""
    console.print(Panel.fit(
        "[bold green]Demo 2: Automatic Refinement[/bold green]\n\n"
        "Recursive refinement triggers when quality < threshold (0.85).\n"
        "Strategy is auto-selected based on query characteristics.",
        border_style="green"
    ))

    config = Config.fast()
    shards = create_test_shards()

    # Create orchestrator with auto-refinement
    orchestrator = RecursiveWeavingOrchestrator(
        cfg=config,
        shards=shards,
        enable_recursive=True,
        quality_threshold=0.85,  # Trigger refinement if < 0.85
        max_iterations=3,
        default_strategy=ReasoningStrategy.ADAPTIVE  # Auto-select strategy
    )

    query = Query(text="Explain how Thompson Sampling balances exploration and exploitation")

    spacetime = await orchestrator.weave(query)

    console.print(f"\n[bold]Query:[/bold] {query.text}")
    console.print(f"[bold]Response:[/bold] {spacetime.response}")
    console.print(f"[bold]Confidence:[/bold] {spacetime.confidence:.2f}")
    console.print(f"[bold]Iterations:[/bold] {spacetime.iterations}")
    console.print(f"[bold]Strategy:[/bold] {spacetime.strategy_used.value if spacetime.strategy_used else 'adaptive'}")

    # Show reasoning provenance
    if spacetime.reasoning_journal:
        console.print("\n[bold]Reasoning Provenance:[/bold]")
        for trace in spacetime.reasoning_journal.traces:
            console.print(f"  Iteration {trace.iteration}: confidence={trace.confidence:.2f}")
            console.print(f"    Thought: {trace.thought[:80]}...")
            console.print(f"    Action: {trace.action}")


async def demo_3_explicit_strategies():
    """Demo 3: Explicit recursive strategies"""
    console.print(Panel.fit(
        "[bold magenta]Demo 3: Explicit Strategies[/bold magenta]\n\n"
        "Compare different recursive reasoning strategies:\n"
        "• REFINE: Iterative improvement\n"
        "• CRITIQUE: Self-critique loop\n"
        "• DECOMPOSE: Break down → solve → synthesize\n"
        "• EXPLORE: Multiple approaches → select best\n"
        "• HOFSTADTER: Meta-reasoning",
        border_style="magenta"
    ))

    config = Config.fast()
    shards = create_test_shards()

    orchestrator = RecursiveWeavingOrchestrator(
        cfg=config,
        shards=shards,
        enable_recursive=True
    )

    query = Query(text="What is consciousness?")

    # Compare strategies
    strategies = [
        ReasoningStrategy.REFINE,
        ReasoningStrategy.CRITIQUE,
        ReasoningStrategy.DECOMPOSE,
        ReasoningStrategy.HOFSTADTER
    ]

    results = []
    for strategy in strategies:
        console.print(f"\n[bold]Testing strategy: {strategy.value}[/bold]")

        spacetime = await orchestrator.weave_with_strategy(
            query=query,
            strategy=strategy,
            max_iterations=2  # Reduced for demo
        )

        results.append({
            "strategy": strategy.value,
            "iterations": spacetime.iterations,
            "confidence": spacetime.confidence,
            "response_length": len(spacetime.response)
        })

        console.print(f"  Iterations: {spacetime.iterations}")
        console.print(f"  Confidence: {spacetime.confidence:.2f}")
        console.print(f"  Response: {spacetime.response[:100]}...")

    # Summary table
    console.print("\n[bold]Strategy Comparison:[/bold]")
    table = Table(title="Recursive Strategy Performance")
    table.add_column("Strategy", style="cyan")
    table.add_column("Iterations", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Response Length", justify="right")

    for result in results:
        table.add_row(
            result["strategy"],
            str(result["iterations"]),
            f"{result['confidence']:.2f}",
            str(result["response_length"])
        )

    console.print(table)


async def demo_4_reasoning_journal():
    """Demo 4: Complete reasoning provenance"""
    console.print(Panel.fit(
        "[bold yellow]Demo 4: Reasoning Provenance[/bold yellow]\n\n"
        "Full scratchpad reasoning trace showing thought process.",
        border_style="yellow"
    ))

    config = Config.fast()
    shards = create_test_shards()

    orchestrator = RecursiveWeavingOrchestrator(
        cfg=config,
        shards=shards,
        enable_recursive=True
    )

    query = Query(text="Explain recursive loops")

    spacetime = await orchestrator.weave_with_strategy(
        query=query,
        strategy=ReasoningStrategy.CRITIQUE,
        max_iterations=3
    )

    if spacetime.reasoning_journal:
        console.print("\n[bold]Full Reasoning History:[/bold]")
        console.print(Markdown(spacetime.reasoning_journal.get_history()))

        console.print("\n[bold]Confidence Trajectory:[/bold]")
        trajectory = spacetime.reasoning_journal.get_confidence_trajectory()
        for i, conf in enumerate(trajectory, 1):
            bar = "█" * int(conf * 50)
            console.print(f"  Iteration {i}: {bar} {conf:.2f}")


async def demo_5_hofstadter_meta_reasoning():
    """Demo 5: Hofstadter strange loops"""
    console.print(Panel.fit(
        "[bold red]Demo 5: Hofstadter Meta-Reasoning[/bold red]\n\n"
        "Strange loops for meta-cognitive questions that require\n"
        "reasoning about reasoning itself.",
        border_style="red"
    ))

    config = Config.fast()
    shards = create_test_shards()

    orchestrator = RecursiveWeavingOrchestrator(
        cfg=config,
        shards=shards,
        enable_recursive=True
    )

    # Meta-cognitive question
    query = Query(text="What does it mean to understand understanding?")

    spacetime = await orchestrator.weave_with_strategy(
        query=query,
        strategy=ReasoningStrategy.HOFSTADTER,
        max_iterations=3
    )

    console.print(f"\n[bold]Meta-Cognitive Query:[/bold] {query.text}")
    console.print(f"\n[bold]Response:[/bold]")
    console.print(Panel(spacetime.response, border_style="red"))

    console.print(f"\n[bold]Iterations:[/bold] {spacetime.iterations}")
    console.print(f"[bold]Final Confidence:[/bold] {spacetime.confidence:.2f}")

    if spacetime.reasoning_journal:
        console.print("\n[bold]Meta-Reasoning Trace:[/bold]")
        for trace in spacetime.reasoning_journal.traces:
            console.print(f"\n  [cyan]Iteration {trace.iteration}[/cyan]")
            console.print(f"  {trace.thought}")


async def main():
    """Run all demos"""
    console.print(Panel.fit(
        "[bold white on blue] Promptly → HoloLoom Integration Demo [/bold white on blue]\n\n"
        "[italic]Showcasing recursive intelligence integrated into HoloLoom's weaving architecture[/italic]",
        border_style="blue"
    ))

    demos = [
        ("Basic Weaving", demo_1_basic_weaving),
        ("Automatic Refinement", demo_2_automatic_refinement),
        ("Explicit Strategies", demo_3_explicit_strategies),
        ("Reasoning Provenance", demo_4_reasoning_journal),
        ("Hofstadter Meta-Reasoning", demo_5_hofstadter_meta_reasoning)
    ]

    for i, (name, demo_fn) in enumerate(demos, 1):
        console.print(f"\n\n{'='*70}")
        console.print(f"[bold]Demo {i}/{len(demos)}: {name}[/bold]")
        console.print(f"{'='*70}\n")

        try:
            await demo_fn()
        except Exception as e:
            console.print(f"[red]Error in {name}: {e}[/red]")
            import traceback
            traceback.print_exc()

        if i < len(demos):
            console.print("\n[dim]Press Enter to continue...[/dim]")
            input()

    console.print("\n\n" + "="*70)
    console.print("[bold green]✅ All demos complete![/bold green]")
    console.print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
