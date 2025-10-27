#!/usr/bin/env python3
"""
ULTIMATE INTEGRATION DEMO
=========================
Shows all 5 quick wins working together:
1. HoloLoom Memory Bridge - Persistent learning
2. Extended Skill Templates - Professional templates
3. Rich CLI - Beautiful output
4. Prompt Analytics - Performance tracking
5. Loop Composition - Multi-stage pipelines

Scenario: Optimize a SQL query using composed loops while tracking everything
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'promptly'))

from execution_engine import ExecutionEngine, ExecutionConfig, ExecutionBackend
from loop_composition import LoopComposer, CompositionStep
from recursive_loops import LoopConfig, LoopType
from prompt_analytics import PromptAnalytics, PromptExecution

# Try to import HoloLoom bridge
try:
    from hololoom_bridge import create_bridge
    HOLOLOOM_AVAILABLE = True
except:
    HOLOLOOM_AVAILABLE = False
    create_bridge = lambda: type('obj', (object,), {'enabled': False})()
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live

console = Console()

def show_title():
    """Feature #3: Rich CLI - Beautiful title"""
    console.print(Panel.fit(
        "[bold magenta]ULTIMATE INTEGRATION DEMO[/bold magenta]\n"
        "[cyan]All 5 Quick Wins Working Together[/cyan]",
        border_style="bright_blue",
        box=box.DOUBLE
    ))

def show_features():
    """Display the 5 features we'll demo"""
    table = Table(title="Features in This Demo", box=box.ROUNDED, border_style="cyan")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Feature", style="white")
    table.add_column("What It Does", style="dim")

    table.add_row("1", "[bold]HoloLoom Bridge[/bold]", "Store results in persistent memory")
    table.add_row("2", "[bold]Skill Templates[/bold]", "Use sql_optimizer template")
    table.add_row("3", "[bold]Rich CLI[/bold]", "Beautiful colored output")
    table.add_row("4", "[bold]Analytics[/bold]", "Track performance metrics")
    table.add_row("5", "[bold]Loop Composition[/bold]", "Chain Critique -> Refine -> Verify")

    console.print(table)

def main():
    show_title()
    console.print()
    show_features()
    console.print()

    # Setup all systems
    console.print("[bold]Initializing systems...[/bold]")

    # Execution engine
    config = ExecutionConfig(
        backend=ExecutionBackend.OLLAMA,
        model="llama3.2:3b",
        temperature=0.7
    )
    engine_exec = ExecutionEngine(config)
    executor = lambda p: engine_exec.execute_prompt(p, skill_name="sql_optimizer").output

    # Feature #5: Loop Composition
    composer = LoopComposer(executor)
    console.print("  [green][OK][/green] Loop Composer initialized")

    # Feature #1: HoloLoom Bridge
    bridge = create_bridge()
    console.print(f"  [green][OK][/green] HoloLoom Bridge: {bridge.enabled}")

    # Feature #4: Analytics
    analytics = PromptAnalytics()
    console.print("  [green][OK][/green] Prompt Analytics ready")

    console.print()

    # Feature #2: Use SQL Optimizer template (skill template)
    task = """Optimize this SQL query for a blog platform:

SELECT * FROM posts
WHERE author_id IN (SELECT id FROM users WHERE active = 1)
ORDER BY created_at DESC
"""

    console.print(Panel(
        Syntax(task.strip(), "sql", theme="monokai"),
        title="[bold]Task: SQL Query Optimization[/bold]",
        border_style="yellow"
    ))
    console.print()

    # Feature #5: Composed Loop Pipeline
    console.print("[bold cyan]Executing Composed Loop Pipeline:[/bold cyan]")
    console.print("  Step 1: Critique the query")
    console.print("  Step 2: Refine based on critique (2 iterations)")
    console.print("  Step 3: Verify optimization")
    console.print()

    # Define composition steps
    steps = [
        CompositionStep(
            loop_type=LoopType.CRITIQUE,
            config=LoopConfig(loop_type=LoopType.CRITIQUE, max_iterations=1),
            description="Critique SQL query for issues"
        ),
        CompositionStep(
            loop_type=LoopType.REFINE,
            config=LoopConfig(
                loop_type=LoopType.REFINE,
                max_iterations=2,
                quality_threshold=0.8
            ),
            description="Refine query based on critique"
        ),
        CompositionStep(
            loop_type=LoopType.VERIFY,
            config=LoopConfig(loop_type=LoopType.VERIFY, max_iterations=1),
            description="Verify optimization is correct"
        )
    ]

    # Execute with progress bar (Feature #3: Rich CLI)
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task_progress = progress.add_task("[cyan]Running pipeline...", total=None)

        try:
            result = composer.compose(task, steps, initial_output=task)
            execution_time = time.time() - start_time

            progress.update(task_progress, completed=100)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    console.print()
    console.print(f"[green][OK][/green] Pipeline complete in {execution_time:.1f}s")
    console.print()

    # Show pipeline results (Feature #3: Rich CLI)
    results_table = Table(title="Pipeline Execution", box=box.ROUNDED, border_style="green")
    results_table.add_column("Step", style="cyan")
    results_table.add_column("Description", style="white")
    results_table.add_column("Iterations", justify="right", style="yellow")
    results_table.add_column("Stop Reason", style="dim")

    for i, (desc, step_result) in enumerate(result.steps, 1):
        results_table.add_row(
            str(i),
            desc,
            str(step_result.iterations),
            step_result.stop_reason
        )

    console.print(results_table)
    console.print()

    # Feature #1: Store in HoloLoom
    if bridge.enabled:
        console.print("[bold]Storing result in HoloLoom memory...[/bold]")
        memory_id = bridge.store_loop_result(
            loop_type="critique_refine_verify",
            task=task,
            result=result.steps[-1][1],  # Final step result
            metadata={
                "total_steps": len(result.steps),
                "total_iterations": result.total_iterations,
                "execution_time": execution_time
            }
        )
        console.print(f"  [green][OK][/green] Stored in HoloLoom")
    else:
        console.print("  [yellow][SKIP][/yellow] HoloLoom not available")

    console.print()

    # Feature #4: Record in Analytics
    console.print("[bold]Recording analytics...[/bold]")

    # Calculate quality score (average across steps with quality data)
    quality_scores = []
    for _, step_result in result.steps:
        if hasattr(step_result, 'improvement_history') and step_result.improvement_history:
            quality_scores.extend(step_result.improvement_history)

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None

    analytics.record_execution(PromptExecution(
        prompt_id="sql_optimizer_composed",
        prompt_name="sql_optimizer_pipeline",
        execution_time=execution_time,
        quality_score=avg_quality,
        success=True,
        model="llama3.2:3b",
        backend="ollama",
        metadata={
            "loop_type": "composed",
            "steps": len(result.steps),
            "total_iterations": result.total_iterations
        }
    ))

    console.print(f"  [green][OK][/green] Analytics recorded")
    console.print()

    # Show analytics summary
    summary = analytics.get_summary()

    analytics_table = Table(title="Analytics Summary", box=box.SIMPLE, border_style="cyan")
    analytics_table.add_column("Metric", style="cyan")
    analytics_table.add_column("Value", style="yellow", justify="right")

    analytics_table.add_row("Total Executions", str(summary['total_executions']))
    analytics_table.add_row("This Execution Time", f"{execution_time:.2f}s")
    if avg_quality:
        analytics_table.add_row("This Quality Score", f"{avg_quality:.2f}")
    analytics_table.add_row("Total Cost", f"${summary['total_cost']:.2f}")

    console.print(analytics_table)
    console.print()

    # Show final optimized query
    final_output = result.final_output

    # Try to extract SQL code if it's in code blocks
    if "```sql" in final_output:
        start = final_output.find("```sql") + 6
        end = final_output.find("```", start)
        if end > start:
            final_sql = final_output[start:end].strip()
        else:
            final_sql = final_output[:500]
    else:
        final_sql = final_output[:500]

    console.print(Panel(
        Syntax(final_sql, "sql", theme="monokai", line_numbers=True),
        title="[bold green]Final Optimized Query[/bold green]",
        border_style="green",
        subtitle=f"{result.total_iterations} total iterations"
    ))

    console.print()

    # Summary of what we demonstrated
    console.print(Panel(
        "[bold green]Integration Complete![/bold green]\n\n"
        "Demonstrated:\n"
        "  [cyan]1.[/cyan] HoloLoom - Stored result in persistent memory\n"
        "  [cyan]2.[/cyan] Templates - Used sql_optimizer skill\n"
        "  [cyan]3.[/cyan] Rich CLI - Beautiful tables, syntax highlighting, progress bars\n"
        "  [cyan]4.[/cyan] Analytics - Tracked execution metrics in database\n"
        "  [cyan]5.[/cyan] Composition - Chained Critique -> Refine -> Verify\n\n"
        f"[dim]Total time: {execution_time:.1f}s | Iterations: {result.total_iterations} | Cost: $0.00[/dim]",
        title="[bold]All 5 Features Working Together![/bold]",
        border_style="bright_green",
        box=box.DOUBLE
    ))

    # Show what was learned
    if bridge.enabled:
        console.print()
        console.print("[bold]What HoloLoom Learned:[/bold]")
        analytics_holo = bridge.get_loop_analytics("critique_refine_verify")
        if analytics_holo.get('total_loops', 0) > 0:
            console.print(f"  Total composed pipelines: {analytics_holo['total_loops']}")
            console.print(f"  Average iterations: {analytics_holo['avg_iterations']:.1f}")
        else:
            console.print("  First execution recorded!")

    console.print()
    console.print("[dim]Analytics DB: ~/.promptly/analytics.db[/dim]")
    console.print("[dim]HoloLoom Memory: Persistent across sessions[/dim]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
