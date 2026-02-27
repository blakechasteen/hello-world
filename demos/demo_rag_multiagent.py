"""
Demo: Multi-Agent RAG with Consensus

Visual demonstration of multi-agent consensus-based query processing showing:
- Simple query with 5 agents, different strategies
- Show all agent responses side-by-side
- Consensus process visualization
- Agreement scores and disagreement detection
- Performance comparison (1 agent vs 5 agents)
- Different consensus methods comparison

Usage:
    python demos/demo_rag_multiagent.py

Author: Agent J (Claude Code)
Date: November 13, 2025
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.config import Config
from hololoom.rag.multiagent_rag import MultiAgentRAG
from hololoom.rag.simple_rag import SimpleRAG

# Rich terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.columns import Columns
    from rich.text import Text
    from rich.layout import Layout
    from rich import box
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠ Rich not available. Install with: pip install rich")
    print("Falling back to plain output.\n")


# ============================================================================
# Demo Configuration
# ============================================================================

DEMO_QUESTIONS = [
    "What is Thompson Sampling?",
    "How does exploration-exploitation tradeoff work?",
    "What are the advantages of Bayesian methods?",
]

DEMO_KNOWLEDGE = [
    "Thompson Sampling is a Bayesian exploration strategy for the multi-armed bandit problem.",
    "Thompson Sampling balances exploration and exploitation by sampling from posterior distributions.",
    "The exploration-exploitation tradeoff is central to reinforcement learning and decision making.",
    "Bayesian methods provide a principled framework for incorporating prior knowledge.",
    "Thompson Sampling has near-optimal regret bounds in many bandit settings.",
]


# ============================================================================
# Rich Visualizations
# ============================================================================

def create_agent_panel(agent_response, rank: int = None):
    """Create Rich panel for single agent response."""
    if not RICH_AVAILABLE:
        return f"Agent {agent_response.agent_id}: {agent_response.response[:50]}..."

    # Status color
    if agent_response.error:
        status_color = "red"
        status_symbol = "✗"
    else:
        status_color = "green"
        status_symbol = "✓"

    # Confidence color
    if agent_response.confidence > 0.8:
        conf_color = "green"
    elif agent_response.confidence > 0.6:
        conf_color = "yellow"
    else:
        conf_color = "red"

    # Build content
    rank_text = f"Rank #{rank} " if rank else ""
    title = f"[bold]{rank_text}{agent_response.agent_id}[/bold] [{status_color}]{status_symbol}[/{status_color}]"

    content = []
    content.append(f"[dim]Strategy:[/dim] {agent_response.strategy}")
    content.append(f"[dim]Confidence:[/dim] [{conf_color}]{agent_response.confidence:.2f}[/{conf_color}]")
    content.append(f"[dim]Latency:[/dim] {agent_response.latency_ms:.1f}ms")

    if agent_response.error:
        content.append(f"\n[red]Error: {agent_response.error}[/red]")
    else:
        content.append(f"\n[dim]Response:[/dim]")
        content.append(agent_response.response[:200] + ("..." if len(agent_response.response) > 200 else ""))
        content.append(f"\n[dim]Sources:[/dim] {len(agent_response.sources)}")

    return Panel(
        "\n".join(content),
        title=title,
        border_style=status_color,
        padding=(0, 1),
    )


def create_consensus_panel(result):
    """Create Rich panel showing consensus result."""
    if not RICH_AVAILABLE:
        return f"Consensus: {result.response[:100]}..."

    # Agreement color
    if result.agreement_score > 0.8:
        agreement_color = "green"
    elif result.agreement_score > 0.6:
        agreement_color = "yellow"
    else:
        agreement_color = "red"

    content = []
    content.append(f"[bold]Consensus Method:[/bold] {result.consensus_method}")
    content.append(f"[bold]Agreement Score:[/bold] [{agreement_color}]{result.agreement_score:.2f}[/{agreement_color}]")
    content.append(f"[bold]Confidence:[/bold] {result.confidence:.2f}")
    content.append("")
    content.append(f"[dim]Response:[/dim]")
    content.append(result.response)
    content.append("")
    content.append(f"[dim]Sources:[/dim] {len(result.sources)} unique sources")

    # Add disagreements if present
    if result.disagreements:
        content.append("")
        content.append("[yellow]⚠ Disagreements Detected:[/yellow]")
        for disagreement in result.disagreements[:3]:
            content.append(f"  • {disagreement}")

    return Panel(
        "\n".join(content),
        title="[bold green]Consensus Result[/bold green]",
        border_style="green",
        padding=(1, 2),
    )


def create_performance_table(result):
    """Create Rich table showing performance metrics."""
    if not RICH_AVAILABLE:
        return "Performance metrics not available (install rich)"

    table = Table(title="Performance Metrics", box=box.ROUNDED)

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    meta = result.consensus_metadata
    table.add_row("Total Agents", str(meta['n_agents']))
    table.add_row("Successful", f"[green]{meta['n_successful']}[/green]")
    table.add_row("Failed", f"[red]{meta['n_failed']}[/red]")
    table.add_row("Consensus Time", f"{meta['consensus_time_ms']:.1f}ms")
    table.add_row("Avg Agent Latency", f"{meta['avg_agent_latency_ms']:.1f}ms")
    table.add_row("Max Agent Latency", f"{meta['max_agent_latency_ms']:.1f}ms")
    table.add_row("Speedup", f"{meta['avg_agent_latency_ms'] / meta['consensus_time_ms']:.1f}×")

    return table


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_simple_query(console):
    """Demo 1: Simple query with 5 agents."""
    console.print("\n[bold cyan]Demo 1: Simple Multi-Agent Query[/bold cyan]\n")

    # Create multi-agent RAG
    async with MultiAgentRAG(
        n_agents=5,
        consensus_method="confidence_weighted",
        agent_timeout=30.0,
    ) as rag:
        # Ingest knowledge
        console.print("[dim]Ingesting knowledge...[/dim]")
        for knowledge in DEMO_KNOWLEDGE:
            await rag.ingest(knowledge)

        # Query
        question = DEMO_QUESTIONS[0]
        console.print(f"\n[bold]Question:[/bold] {question}\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Querying agents...", total=None)

            result = await rag.query_multiagent(question, explain_disagreement=True)

            progress.update(task, completed=True)

        # Show agent responses
        console.print("\n[bold]Agent Responses:[/bold]\n")

        # Sort by confidence
        sorted_responses = sorted(
            result.agent_responses,
            key=lambda r: r.confidence if r.error is None else 0.0,
            reverse=True
        )

        # Create panels for each agent
        panels = []
        for i, resp in enumerate(sorted_responses, 1):
            panels.append(create_agent_panel(resp, rank=i))

        # Display in columns
        console.print(Columns(panels[:3], equal=True, expand=True))
        console.print(Columns(panels[3:], equal=True, expand=True))

        # Show consensus
        console.print("\n")
        console.print(create_consensus_panel(result))

        # Show performance
        console.print("\n")
        console.print(create_performance_table(result))

        return result


async def demo_consensus_methods(console):
    """Demo 2: Compare different consensus methods."""
    console.print("\n\n[bold cyan]Demo 2: Consensus Methods Comparison[/bold cyan]\n")

    question = DEMO_QUESTIONS[1]
    console.print(f"[bold]Question:[/bold] {question}\n")

    results = {}

    for method in ["majority_vote", "confidence_weighted", "ensemble"]:
        console.print(f"\n[dim]Testing {method}...[/dim]")

        async with MultiAgentRAG(
            n_agents=5,
            consensus_method=method,
        ) as rag:
            # Ingest knowledge
            for knowledge in DEMO_KNOWLEDGE:
                await rag.ingest(knowledge)

            # Query
            result = await rag.query_multiagent(question)
            results[method] = result

    # Create comparison table
    if RICH_AVAILABLE:
        table = Table(title="Consensus Methods Comparison", box=box.ROUNDED)
        table.add_column("Method", style="cyan")
        table.add_column("Agreement", style="magenta")
        table.add_column("Confidence", style="green")
        table.add_column("Time (ms)", style="yellow")

        for method, result in results.items():
            table.add_row(
                method,
                f"{result.agreement_score:.2f}",
                f"{result.confidence:.2f}",
                f"{result.consensus_metadata['consensus_time_ms']:.1f}",
            )

        console.print("\n")
        console.print(table)
    else:
        print("\nConsensus Methods Comparison:")
        for method, result in results.items():
            print(f"  {method}: agreement={result.agreement_score:.2f}, confidence={result.confidence:.2f}")


async def demo_single_vs_multi(console):
    """Demo 3: Single agent vs multi-agent comparison."""
    console.print("\n\n[bold cyan]Demo 3: Single Agent vs Multi-Agent[/bold cyan]\n")

    question = DEMO_QUESTIONS[2]
    console.print(f"[bold]Question:[/bold] {question}\n")

    # Single agent
    console.print("[dim]Testing single agent...[/dim]")
    async with SimpleRAG() as rag:
        for knowledge in DEMO_KNOWLEDGE:
            await rag.ingest(knowledge)

        import time
        start = time.time()
        single_result = await rag.query(question)
        single_time = (time.time() - start) * 1000

    # Multi-agent
    console.print("[dim]Testing multi-agent (5 agents)...[/dim]")
    async with MultiAgentRAG(n_agents=5) as rag:
        for knowledge in DEMO_KNOWLEDGE:
            await rag.ingest(knowledge)

        multi_result = await rag.query_multiagent(question)

    # Comparison
    if RICH_AVAILABLE:
        table = Table(title="Single vs Multi-Agent", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Single Agent", style="magenta")
        table.add_column("Multi-Agent (5)", style="green")

        table.add_row("Confidence", f"{single_result.confidence:.2f}", f"{multi_result.confidence:.2f}")
        table.add_row("Sources", str(len(single_result.sources)), str(len(multi_result.sources)))
        table.add_row("Time", f"{single_time:.1f}ms", f"{multi_result.consensus_metadata['consensus_time_ms']:.1f}ms")
        table.add_row("Agreement", "N/A", f"{multi_result.agreement_score:.2f}")

        console.print("\n")
        console.print(table)
    else:
        print("\nSingle vs Multi-Agent Comparison:")
        print(f"  Single agent: confidence={single_result.confidence:.2f}, time={single_time:.1f}ms")
        print(f"  Multi-agent: confidence={multi_result.confidence:.2f}, time={multi_result.consensus_metadata['consensus_time_ms']:.1f}ms, agreement={multi_result.agreement_score:.2f}")


async def demo_disagreement_detection(console):
    """Demo 4: Disagreement detection."""
    console.print("\n\n[bold cyan]Demo 4: Disagreement Detection[/bold cyan]\n")

    # Use a more ambiguous question to induce disagreement
    question = "What is the best approach to exploration?"
    console.print(f"[bold]Question:[/bold] {question}\n")
    console.print("[dim]This ambiguous question should induce disagreement...[/dim]\n")

    async with MultiAgentRAG(
        n_agents=5,
        consensus_method="confidence_weighted",
    ) as rag:
        # Ingest knowledge
        for knowledge in DEMO_KNOWLEDGE:
            await rag.ingest(knowledge)

        # Query with disagreement explanation
        result = await rag.query_multiagent(question, explain_disagreement=True)

        # Show consensus
        console.print(create_consensus_panel(result))

        # Highlight disagreements
        if result.disagreements:
            console.print("\n[bold yellow]Disagreements Detected:[/bold yellow]\n")
            for i, disagreement in enumerate(result.disagreements, 1):
                console.print(f"  {i}. {disagreement}")
        else:
            console.print("\n[green]No significant disagreements detected[/green]")


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run all demos."""
    if RICH_AVAILABLE:
        console = Console()
        console.print("\n[bold magenta]Multi-Agent RAG Demo[/bold magenta]")
        console.print("[dim]Consensus-based multi-agent query processing[/dim]\n")
    else:
        console = None
        print("\nMulti-Agent RAG Demo")
        print("=" * 50)

    try:
        # Run demos
        await demo_simple_query(console or sys.stdout)
        await demo_consensus_methods(console or sys.stdout)
        await demo_single_vs_multi(console or sys.stdout)
        await demo_disagreement_detection(console or sys.stdout)

        if RICH_AVAILABLE:
            console.print("\n[bold green]✓ All demos complete![/bold green]\n")
        else:
            print("\n✓ All demos complete!\n")

    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[yellow]Demo interrupted by user[/yellow]\n")
        else:
            print("\nDemo interrupted by user\n")
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n[red]Error: {e}[/red]\n")
        else:
            print(f"\nError: {e}\n")
        raise


if __name__ == "__main__":
    asyncio.run(main())
