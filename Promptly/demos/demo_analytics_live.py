#!/usr/bin/env python3
"""
Live Prompt Analytics Demo
Shows analytics system actually working with real loop executions
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'promptly'))

from execution_engine import ExecutionEngine, ExecutionConfig, ExecutionBackend
from recursive_loops import RecursiveEngine, LoopConfig, LoopType
from prompt_analytics import PromptAnalytics, PromptExecution
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def run_demo():
    """Run loops and track analytics"""

    console.print(Panel(
        "[bold cyan]Live Prompt Analytics Demo[/bold cyan]\n"
        "Execute 3 loops and watch analytics update in real-time!",
        border_style="cyan"
    ))

    # Setup
    config = ExecutionConfig(
        backend=ExecutionBackend.OLLAMA,
        model="llama3.2:3b",
        temperature=0.7
    )
    engine_exec = ExecutionEngine(config)
    executor = lambda prompt: engine_exec.execute_prompt(prompt, skill_name="demo").output

    engine = RecursiveEngine(executor)
    analytics = PromptAnalytics()

    # Test prompts
    prompts = [
        {
            "name": "simple_question",
            "task": "What is 2+2?",
            "loop_type": LoopType.REFINE,
            "config": LoopConfig(loop_type=LoopType.REFINE, max_iterations=2)
        },
        {
            "name": "explain_concept",
            "task": "Explain recursion briefly",
            "loop_type": LoopType.REFINE,
            "config": LoopConfig(loop_type=LoopType.REFINE, max_iterations=3, quality_threshold=0.8)
        },
        {
            "name": "meta_thinking",
            "task": "What is thinking?",
            "loop_type": LoopType.HOFSTADTER,
            "config": LoopConfig(loop_type=LoopType.HOFSTADTER, max_iterations=2)
        }
    ]

    console.print("\n[bold]Executing prompts and tracking analytics...[/bold]\n")

    for i, prompt_config in enumerate(prompts, 1):
        console.print(f"[cyan]{i}/3[/cyan] Running '{prompt_config['name']}'...")

        start_time = time.time()

        try:
            # Execute loop
            if prompt_config['loop_type'] == LoopType.REFINE:
                result = engine.execute_refine_loop(
                    task=prompt_config['task'],
                    initial_output=f"Answer to: {prompt_config['task']}",
                    config=prompt_config['config']
                )
            else:
                result = engine.execute_hofstadter_loop(
                    task=prompt_config['task'],
                    config=prompt_config['config']
                )

            execution_time = time.time() - start_time

            # Calculate quality score
            quality = None
            if result.improvement_history:
                quality = sum(result.improvement_history) / len(result.improvement_history)

            # Record in analytics
            analytics.record_execution(PromptExecution(
                prompt_id=f"{prompt_config['name']}_{i}",
                prompt_name=prompt_config['name'],
                execution_time=execution_time,
                quality_score=quality,
                success=result.success,
                model="llama3.2:3b",
                backend="ollama",
                tokens_used=None,  # Ollama doesn't report tokens easily
                cost=0.0,  # Free!
                metadata={
                    "loop_type": prompt_config['loop_type'].value,
                    "iterations": result.iterations,
                    "stop_reason": result.stop_reason
                }
            ))

            console.print(f"  [green][OK][/green] Complete in {execution_time:.1f}s")
            console.print(f"  Iterations: {result.iterations}, Stop: {result.stop_reason}")
            if quality:
                console.print(f"  Quality: {quality:.2f}")

        except Exception as e:
            console.print(f"  [red][FAIL][/red] Error: {e}")

            # Record failure
            analytics.record_execution(PromptExecution(
                prompt_id=f"{prompt_config['name']}_{i}_failed",
                prompt_name=prompt_config['name'],
                execution_time=time.time() - start_time,
                success=False,
                model="llama3.2:3b",
                backend="ollama"
            ))

        console.print()

    # Show analytics
    console.print("\n" + "="*70)
    console.print("[bold green]ANALYTICS RESULTS[/bold green]")
    console.print("="*70 + "\n")

    # Summary
    summary = analytics.get_summary()

    summary_table = Table(title="Overall Summary", box=box.ROUNDED, border_style="green")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="yellow", justify="right")

    summary_table.add_row("Total Executions", str(summary['total_executions']))
    summary_table.add_row("Unique Prompts", str(summary['unique_prompts']))
    summary_table.add_row("Success Rate", f"{summary['success_rate']:.1%}")
    summary_table.add_row("Avg Execution Time", f"{summary['avg_execution_time']:.2f}s")
    if summary.get('avg_quality_score'):
        summary_table.add_row("Avg Quality Score", f"{summary['avg_quality_score']:.2f}")
    summary_table.add_row("Total Cost", f"${summary['total_cost']:.2f}")

    console.print(summary_table)
    console.print()

    # Per-prompt stats
    stats_table = Table(title="Per-Prompt Statistics", box=box.ROUNDED, border_style="cyan")
    stats_table.add_column("Prompt Name", style="cyan")
    stats_table.add_column("Executions", justify="right")
    stats_table.add_column("Success Rate", justify="right")
    stats_table.add_column("Avg Time", justify="right")
    stats_table.add_column("Avg Quality", justify="right")
    stats_table.add_column("Trend", style="yellow")

    for prompt_name in ['simple_question', 'explain_concept', 'meta_thinking']:
        stats = analytics.get_prompt_stats(prompt_name)
        if stats:
            stats_table.add_row(
                stats.prompt_name,
                str(stats.total_executions),
                f"{stats.success_rate:.1%}",
                f"{stats.avg_execution_time:.2f}s",
                f"{stats.avg_quality_score:.2f}" if stats.avg_quality_score else "N/A",
                stats.trend
            )

    console.print(stats_table)
    console.print()

    # Recommendations
    recs = analytics.get_recommendations()

    if recs:
        console.print(Panel(
            "\n".join(f"• {rec}" for rec in recs),
            title="[bold]Recommendations[/bold]",
            border_style="yellow"
        ))

    # Top performers
    console.print()
    top_by_speed = analytics.get_top_prompts(metric="speed", limit=3)

    if top_by_speed:
        top_table = Table(title="Fastest Prompts", box=box.SIMPLE)
        top_table.add_column("Prompt", style="green")
        top_table.add_column("Avg Time", justify="right", style="yellow")

        for stats in top_by_speed:
            top_table.add_row(stats.prompt_name, f"{stats.avg_execution_time:.2f}s")

        console.print(top_table)

    console.print("\n[dim]Analytics database: ~/.promptly/analytics.db[/dim]")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
