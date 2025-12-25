"""
Demo 5: HoloLoom Integration

Demonstrates Promptly's integration with HoloLoom memory system:
- Storing prompt executions in memory
- Thompson Sampling recommendations
- Learning insights per task type
- Similar prompt retrieval
- Quality tracking over time

Interactive flow:
[1] Store a prompt execution
[2] Get prompt recommendation
[3] View learning insights
[4] Find similar prompts
[5] View execution statistics
[Q] Return to main menu
"""

import asyncio
from typing import TYPE_CHECKING, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import random

if TYPE_CHECKING:
    from .interactive_demo import DemoRunner


# Simulated execution history for demo
@dataclass
class ExecutionRecord:
    """Record of a prompt execution."""
    prompt_name: str
    task_type: str
    quality_score: float
    latency_ms: float
    timestamp: str
    context: Dict[str, Any]


# Sample execution history
SAMPLE_EXECUTIONS: List[ExecutionRecord] = [
    ExecutionRecord(
        prompt_name="summarize",
        task_type="text_processing",
        quality_score=0.92,
        latency_ms=450,
        timestamp="2025-01-15T10:30:00",
        context={"input_length": 2500, "output_length": 150}
    ),
    ExecutionRecord(
        prompt_name="code-review",
        task_type="code_analysis",
        quality_score=0.88,
        latency_ms=680,
        timestamp="2025-01-15T11:15:00",
        context={"language": "Python", "lines": 120}
    ),
    ExecutionRecord(
        prompt_name="explain",
        task_type="education",
        quality_score=0.95,
        latency_ms=320,
        timestamp="2025-01-15T14:00:00",
        context={"audience": "beginner", "topic": "recursion"}
    ),
    ExecutionRecord(
        prompt_name="summarize",
        task_type="text_processing",
        quality_score=0.89,
        latency_ms=520,
        timestamp="2025-01-16T09:00:00",
        context={"input_length": 4000, "output_length": 200}
    ),
    ExecutionRecord(
        prompt_name="code-review",
        task_type="code_analysis",
        quality_score=0.91,
        latency_ms=750,
        timestamp="2025-01-16T10:30:00",
        context={"language": "TypeScript", "lines": 85}
    ),
]

# Thompson Sampling priors (alpha, beta) for each prompt
THOMPSON_PRIORS: Dict[str, tuple] = {
    "summarize": (12, 3),      # High success rate
    "code-review": (8, 2),    # Good success rate
    "explain": (15, 2),       # Excellent success rate
    "translate": (5, 3),      # Moderate success rate
    "analyze": (6, 4),        # Lower success rate
}

# Task type insights
TASK_INSIGHTS: Dict[str, Dict[str, Any]] = {
    "text_processing": {
        "best_prompt": "summarize",
        "avg_quality": 0.90,
        "total_executions": 45,
        "success_rate": 0.89,
        "common_contexts": ["document analysis", "report generation"],
    },
    "code_analysis": {
        "best_prompt": "code-review",
        "avg_quality": 0.87,
        "total_executions": 32,
        "success_rate": 0.84,
        "common_contexts": ["PR review", "security audit"],
    },
    "education": {
        "best_prompt": "explain",
        "avg_quality": 0.93,
        "total_executions": 28,
        "success_rate": 0.92,
        "common_contexts": ["tutorials", "documentation"],
    },
}


async def run_hololoom_demo(runner: "DemoRunner"):
    """
    Run the HoloLoom integration demo.

    Args:
        runner: DemoRunner instance for UI
    """
    runner.print_info("Welcome to HoloLoom Integration!")
    runner.print()
    runner.print("[dim]HoloLoom provides memory and learning capabilities for Promptly.[/dim]")
    runner.print("[dim]Store executions, get recommendations, and learn from outcomes.[/dim]")
    runner.print()

    # Initialize demo state
    executions = list(SAMPLE_EXECUTIONS)
    priors = dict(THOMPSON_PRIORS)

    while True:
        # Show demo menu
        runner.print()
        runner.print_table(
            "🧠 HoloLoom Integration Demo",
            ["Option", "Action"],
            [
                ["1", "Store a prompt execution"],
                ["2", "Get prompt recommendation"],
                ["3", "View learning insights"],
                ["4", "Find similar prompts"],
                ["5", "View execution statistics"],
                ["Q", "Return to main menu"],
            ]
        )

        choice = runner.prompt_choice("Select action", ["1", "2", "3", "4", "5", "q", "Q"], "1")

        if choice.lower() == 'q':
            break

        try:
            if choice == "1":
                await demo_store_execution(runner, executions, priors)
            elif choice == "2":
                await demo_get_recommendation(runner, priors)
            elif choice == "3":
                await demo_view_insights(runner)
            elif choice == "4":
                await demo_find_similar(runner)
            elif choice == "5":
                await demo_view_statistics(runner, executions)
        except Exception as e:
            runner.print_error(f"Error: {e}")


async def demo_store_execution(runner: "DemoRunner", executions: List[ExecutionRecord], priors: Dict[str, tuple]):
    """Demonstrate storing a prompt execution."""
    runner.print()
    runner.print_info("Storing a Prompt Execution")
    runner.print()

    runner.print("[dim]When you run a prompt, the execution is stored in HoloLoom memory.[/dim]")
    runner.print("[dim]This enables learning, recommendations, and quality tracking.[/dim]")
    runner.print()

    # Get execution details
    prompt_name = runner.prompt_input("Prompt name", "summarize")
    task_type = runner.prompt_input("Task type", "text_processing")
    quality = runner.prompt_input("Quality score (0.0-1.0)", "0.85")

    try:
        quality_score = float(quality)
        quality_score = max(0.0, min(1.0, quality_score))
    except ValueError:
        quality_score = 0.85

    # Simulate storing
    with runner.show_progress("Storing execution") as progress:
        task = progress.add_task("Processing...", total=3)

        # Step 1: Store in memory
        progress.update(task, advance=1, description="Storing in memory...")
        await asyncio.sleep(0.3)

        # Step 2: Update Thompson priors
        progress.update(task, advance=1, description="Updating priors...")
        await asyncio.sleep(0.2)

        # Update the prior
        if prompt_name in priors:
            alpha, beta = priors[prompt_name]
            if quality_score >= 0.7:
                alpha += quality_score
            else:
                beta += (1 - quality_score)
            priors[prompt_name] = (alpha, beta)
        else:
            if quality_score >= 0.7:
                priors[prompt_name] = (2, 1)
            else:
                priors[prompt_name] = (1, 2)

        # Step 3: Index for retrieval
        progress.update(task, advance=1, description="Indexing...")
        await asyncio.sleep(0.2)

    # Create and store execution record
    new_execution = ExecutionRecord(
        prompt_name=prompt_name,
        task_type=task_type,
        quality_score=quality_score,
        latency_ms=random.randint(200, 800),
        timestamp=datetime.now().isoformat(),
        context={"demo": True}
    )
    executions.append(new_execution)

    runner.print_success("Execution stored successfully!")
    runner.print()

    # Show what was stored
    runner.print("[bold]Stored Record:[/bold]")
    runner.print(f"  Prompt: [cyan]{prompt_name}[/cyan]")
    runner.print(f"  Task Type: {task_type}")
    runner.print(f"  Quality: {quality_score:.2f}")
    runner.print(f"  Timestamp: {new_execution.timestamp}")
    runner.print()

    # Show Thompson update
    alpha, beta = priors[prompt_name]
    expected = alpha / (alpha + beta)
    runner.print("[bold]Thompson Sampling Update:[/bold]")
    runner.print(f"  Prior: α={alpha:.1f}, β={beta:.1f}")
    runner.print(f"  Expected Quality: {expected:.2f}")


async def demo_get_recommendation(runner: "DemoRunner", priors: Dict[str, tuple]):
    """Demonstrate getting a prompt recommendation via Thompson Sampling."""
    runner.print()
    runner.print_info("Getting Prompt Recommendation")
    runner.print()

    runner.print("[dim]Thompson Sampling balances exploration (trying new prompts)[/dim]")
    runner.print("[dim]with exploitation (using proven high-quality prompts).[/dim]")
    runner.print()

    # Get task type
    task_type = runner.prompt_input("Task type", "text_processing")

    # Get available prompts for task type
    available = list(priors.keys())
    runner.print(f"[dim]Available prompts: {', '.join(available)}[/dim]")
    runner.print()

    # Thompson Sampling
    with runner.show_progress("Computing recommendation") as progress:
        task = progress.add_task("Sampling...", total=2)

        # Sample from each prior
        samples = {}
        for name, (alpha, beta) in priors.items():
            # Beta distribution sampling (simulated)
            sample = random.betavariate(alpha, beta)
            samples[name] = sample

        progress.update(task, advance=1, description="Ranking...")
        await asyncio.sleep(0.3)

        # Rank by sample
        ranked = sorted(samples.items(), key=lambda x: -x[1])

        progress.update(task, advance=1, description="Done")
        await asyncio.sleep(0.1)

    # Show recommendation
    recommended = ranked[0][0]
    runner.print_success(f"Recommended prompt: [bold]{recommended}[/bold]")
    runner.print()

    # Show ranking table
    rows = []
    for i, (name, sample) in enumerate(ranked):
        alpha, beta = priors[name]
        expected = alpha / (alpha + beta)
        marker = "→" if i == 0 else " "
        rows.append([
            marker,
            name,
            f"{expected:.2f}",
            f"{sample:.2f}",
            f"α={alpha:.1f}, β={beta:.1f}"
        ])

    runner.print_table(
        "🎯 Thompson Sampling Results",
        ["", "Prompt", "Expected", "Sample", "Prior"],
        rows
    )

    runner.print()
    runner.print("[dim]Explanation:[/dim]")
    runner.print("[dim]• Expected = α/(α+β) - historical quality[/dim]")
    runner.print("[dim]• Sample = random draw from Beta(α,β)[/dim]")
    runner.print("[dim]• Higher α means more successes, higher β means more failures[/dim]")
    runner.print("[dim]• Sampling adds randomness for exploration[/dim]")


async def demo_view_insights(runner: "DemoRunner"):
    """Demonstrate viewing learning insights per task type."""
    runner.print()
    runner.print_info("Learning Insights by Task Type")
    runner.print()

    runner.print("[dim]HoloLoom tracks performance patterns across task types.[/dim]")
    runner.print("[dim]This helps identify which prompts work best for which tasks.[/dim]")
    runner.print()

    # Show task type selection
    task_types = list(TASK_INSIGHTS.keys())
    runner.print(f"[dim]Available task types: {', '.join(task_types)}[/dim]")

    task_type = runner.prompt_input("Task type to analyze", "text_processing")

    if task_type not in TASK_INSIGHTS:
        runner.print_warning(f"No insights for '{task_type}'")
        return

    insights = TASK_INSIGHTS[task_type]

    # Show insights
    runner.print()
    runner.print(f"[bold cyan]Insights for '{task_type}':[/bold cyan]")
    runner.print()

    # Summary stats
    runner.print("[bold]Performance Summary:[/bold]")
    runner.print(f"  Best Prompt: [green]{insights['best_prompt']}[/green]")
    runner.print(f"  Average Quality: {insights['avg_quality']:.2f}")
    runner.print(f"  Success Rate: {insights['success_rate']:.1%}")
    runner.print(f"  Total Executions: {insights['total_executions']}")
    runner.print()

    # Common contexts
    runner.print("[bold]Common Use Cases:[/bold]")
    for ctx in insights['common_contexts']:
        runner.print(f"  • {ctx}")
    runner.print()

    # Quality distribution (simulated)
    runner.print("[bold]Quality Distribution:[/bold]")
    runner.print("  0.9-1.0: " + "█" * 8 + " 40%")
    runner.print("  0.8-0.9: " + "█" * 6 + " 30%")
    runner.print("  0.7-0.8: " + "█" * 4 + " 20%")
    runner.print("  <0.7:    " + "█" * 2 + " 10%")
    runner.print()

    # Recommendations
    runner.print("[bold]Recommendations:[/bold]")
    runner.print(f"  [dim]• Use '{insights['best_prompt']}' for this task type[/dim]")
    runner.print(f"  [dim]• Consider refining prompts with <0.7 quality[/dim]")
    runner.print(f"  [dim]• {insights['total_executions']} executions provide reliable stats[/dim]")


async def demo_find_similar(runner: "DemoRunner"):
    """Demonstrate finding similar prompts via semantic search."""
    runner.print()
    runner.print_info("Finding Similar Prompts")
    runner.print()

    runner.print("[dim]HoloLoom indexes prompt embeddings for semantic search.[/dim]")
    runner.print("[dim]Find prompts similar to a query or existing prompt.[/dim]")
    runner.print()

    # Get search query
    query = runner.prompt_input("Search query", "generate documentation")

    # Simulate search
    with runner.show_progress("Searching") as progress:
        task = progress.add_task("Embedding query...", total=3)
        await asyncio.sleep(0.2)

        progress.update(task, advance=1, description="Searching index...")
        await asyncio.sleep(0.3)

        progress.update(task, advance=1, description="Ranking results...")
        await asyncio.sleep(0.2)

        progress.update(task, advance=1, description="Done")

    # Simulated results based on query
    results = [
        {"name": "code-documenter", "similarity": 0.92, "description": "Generate code documentation"},
        {"name": "explain", "similarity": 0.78, "description": "Explain concepts clearly"},
        {"name": "summarize", "similarity": 0.65, "description": "Summarize text content"},
        {"name": "api-spec-generator", "similarity": 0.61, "description": "Generate API specifications"},
    ]

    runner.print()
    runner.print(f"[bold]Similar prompts for '{query}':[/bold]")
    runner.print()

    # Show results table
    rows = []
    for i, r in enumerate(results, 1):
        sim_bar = "█" * int(r['similarity'] * 10) + "░" * (10 - int(r['similarity'] * 10))
        rows.append([
            str(i),
            r['name'],
            f"{r['similarity']:.2f}",
            sim_bar,
            r['description']
        ])

    runner.print_table(
        "🔍 Search Results",
        ["#", "Prompt", "Similarity", "Score", "Description"],
        rows
    )

    runner.print()
    runner.print("[dim]Similarity is computed using cosine distance between embeddings.[/dim]")
    runner.print("[dim]Higher similarity means the prompt is more relevant to your query.[/dim]")


async def demo_view_statistics(runner: "DemoRunner", executions: List[ExecutionRecord]):
    """Demonstrate viewing execution statistics."""
    runner.print()
    runner.print_info("Execution Statistics")
    runner.print()

    runner.print("[dim]Track quality and performance over time.[/dim]")
    runner.print()

    if not executions:
        runner.print_warning("No executions recorded yet")
        return

    # Compute stats
    total = len(executions)
    avg_quality = sum(e.quality_score for e in executions) / total
    avg_latency = sum(e.latency_ms for e in executions) / total
    success_rate = sum(1 for e in executions if e.quality_score >= 0.7) / total

    # By prompt
    by_prompt: Dict[str, List[ExecutionRecord]] = {}
    for e in executions:
        if e.prompt_name not in by_prompt:
            by_prompt[e.prompt_name] = []
        by_prompt[e.prompt_name].append(e)

    # By task type
    by_task: Dict[str, List[ExecutionRecord]] = {}
    for e in executions:
        if e.task_type not in by_task:
            by_task[e.task_type] = []
        by_task[e.task_type].append(e)

    # Overall stats
    runner.print("[bold]Overall Statistics:[/bold]")
    runner.print(f"  Total Executions: {total}")
    runner.print(f"  Average Quality: {avg_quality:.2f}")
    runner.print(f"  Average Latency: {avg_latency:.0f}ms")
    runner.print(f"  Success Rate (≥0.7): {success_rate:.1%}")
    runner.print()

    # By prompt table
    rows = []
    for name, execs in sorted(by_prompt.items()):
        count = len(execs)
        avg_q = sum(e.quality_score for e in execs) / count
        avg_l = sum(e.latency_ms for e in execs) / count
        rows.append([name, str(count), f"{avg_q:.2f}", f"{avg_l:.0f}ms"])

    runner.print_table(
        "📊 By Prompt",
        ["Prompt", "Count", "Avg Quality", "Avg Latency"],
        rows
    )

    runner.print()

    # By task type table
    rows = []
    for task, execs in sorted(by_task.items()):
        count = len(execs)
        avg_q = sum(e.quality_score for e in execs) / count
        rows.append([task, str(count), f"{avg_q:.2f}"])

    runner.print_table(
        "📈 By Task Type",
        ["Task Type", "Count", "Avg Quality"],
        rows
    )

    # Recent executions
    runner.print()
    runner.print("[bold]Recent Executions:[/bold]")
    for e in executions[-5:]:
        quality_bar = "█" * int(e.quality_score * 10) + "░" * (10 - int(e.quality_score * 10))
        runner.print(f"  {e.prompt_name}: {quality_bar} {e.quality_score:.2f}")


# Direct execution for testing
if __name__ == "__main__":
    from .interactive_demo import DemoRunner

    async def main():
        runner = DemoRunner()
        await runner.setup_sample_data()
        await run_hololoom_demo(runner)

    asyncio.run(main())
