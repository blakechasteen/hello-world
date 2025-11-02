#!/usr/bin/env python3
"""
Rich CLI Demo - Beautiful Colored Terminal Output
Shows Promptly recursive intelligence with gorgeous formatting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'promptly'))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
from rich.layout import Layout
from rich import box
from rich.text import Text
import time

from execution_engine import ExecutionEngine, ExecutionConfig, ExecutionBackend
from recursive_loops import RecursiveEngine, LoopConfig, LoopType

# Force UTF-8 encoding for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Initialize Rich console with safe characters for Windows
console = Console(legacy_windows=False, force_terminal=True)

def show_header():
    """Display beautiful header"""
    title = Text()
    title.append("PROMPTLY ", style="bold magenta")
    title.append("Recursive Intelligence Platform", style="bold cyan")

    panel = Panel(
        title,
        border_style="bright_blue",
        box=box.DOUBLE,
        subtitle="Powered by llama3.2:3b (2GB)",
        subtitle_align="right"
    )
    console.print(panel)

def show_menu():
    """Display menu with Rich"""
    table = Table(title="Available Demos", box=box.ROUNDED, border_style="cyan")

    table.add_column("Demo", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Time", style="yellow", justify="right")

    table.add_row(
        "1",
        "[bold]Strange Loop[/bold] - Meta self-explanation",
        "~30s"
    )
    table.add_row(
        "2",
        "[bold]Consciousness[/bold] - 5 meta-levels of philosophy",
        "~40s"
    )
    table.add_row(
        "3",
        "[bold]Code Evolution[/bold] - Simple to Production-ready",
        "~20s"
    )
    table.add_row(
        "4",
        "[bold]Ultimate Meta[/bold] - Hofstadter + Refinement",
        "~60s"
    )

    console.print(table)

def run_hofstadter_demo():
    """Run Hofstadter strange loop with Rich progress"""
    console.print("\n")
    console.print(Panel(
        "[bold cyan]Demo: What is a strange loop?[/bold cyan]\n"
        "Using a strange loop to explain strange loops - perfectly meta!",
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
    loop_config = LoopConfig(
        loop_type=LoopType.HOFSTADTER,
        max_iterations=4,
        enable_scratchpad=True
    )

    # Progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Running meta-levels...", total=4)

        # We'll manually track progress
        result = engine.execute_hofstadter_loop(
            task="What is a strange loop?",
            config=loop_config
        )

        # Simulate progress (actual execution is synchronous)
        for i in range(4):
            time.sleep(0.5)  # Small delay for visual effect
            progress.update(task, advance=1)

    # Display results
    console.print("\n")
    console.print(Panel(
        f"[bold green]✓ Complete![/bold green]\n\n"
        f"[bold]Iterations:[/bold] {result.iterations}\n"
        f"[bold]Stop Reason:[/bold] {result.stop_reason}",
        title="[bold]Results[/bold]",
        border_style="green"
    ))

    # Show final output with markdown
    console.print("\n")
    console.print(Panel(
        Markdown(result.final_output[:500] + "..." if len(result.final_output) > 500 else result.final_output),
        title="[bold cyan]Final Synthesis[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED
    ))

def run_code_demo():
    """Run code improvement with Rich syntax highlighting"""
    console.print("\n")
    console.print(Panel(
        "[bold cyan]Demo: Code Evolution[/bold cyan]\n"
        "Watch code transform from simple → production-ready",
        border_style="cyan"
    ))

    initial_code = """def f(n):
    if n < 2: return n
    return f(n-1) + f(n-2)"""

    # Show initial code with syntax highlighting
    console.print("\n[bold]Initial Code:[/bold]")
    syntax = Syntax(initial_code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)

    # Setup
    config = ExecutionConfig(
        backend=ExecutionBackend.OLLAMA,
        model="llama3.2:3b",
        temperature=0.7
    )
    engine_exec = ExecutionEngine(config)
    executor = lambda prompt: engine_exec.execute_prompt(prompt, skill_name="demo").output

    engine = RecursiveEngine(executor)
    loop_config = LoopConfig(
        loop_type=LoopType.REFINE,
        max_iterations=3,
        quality_threshold=0.85,
        enable_scratchpad=True
    )

    # Progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Refining code...", total=100)

        result = engine.execute_refine_loop(
            task="Transform into production-ready code with type hints, docstrings, memoization, and error handling",
            initial_output=initial_code,
            config=loop_config
        )

        progress.update(task, completed=100)

    # Quality progression
    if result.improvement_history:
        console.print("\n[bold]Quality Progression:[/bold]")

        quality_table = Table(box=box.SIMPLE)
        quality_table.add_column("Iteration", style="cyan")
        quality_table.add_column("Score", style="yellow")
        quality_table.add_column("Progress", style="green")

        for i, score in enumerate(result.improvement_history, 1):
            bar = "█" * int(score * 30)
            quality_table.add_row(
                str(i),
                f"{score:.2f}",
                bar
            )

        console.print(quality_table)

    # Show final code
    console.print(f"\n[bold green]✓ {result.stop_reason}[/bold green]")
    console.print(f"[bold]Iterations:[/bold] {result.iterations}\n")

    # Extract code from final output
    final_code = result.final_output
    if "```python" in final_code:
        start = final_code.find("```python") + 9
        end = final_code.find("```", start)
        if end > start:
            final_code = final_code[start:end].strip()

    console.print("[bold]Final Code:[/bold]")
    syntax = Syntax(final_code[:1000], "python", theme="monokai", line_numbers=True)
    console.print(syntax)

def main():
    """Main demo with Rich"""
    show_header()
    console.print("\n")
    show_menu()

    console.print("\n")
    choice = console.input("[bold cyan]Choose demo (1-4):[/bold cyan] ")

    if choice == "1":
        run_hofstadter_demo()
    elif choice == "2":
        console.print("[yellow]Demo 2 not implemented in this quick demo[/yellow]")
    elif choice == "3":
        run_code_demo()
    elif choice == "4":
        console.print("[yellow]Demo 4 not implemented in this quick demo[/yellow]")
    else:
        console.print("[red]Invalid choice[/red]")
        return

    console.print("\n")
    console.print(Panel(
        "[bold green]Demo Complete![/bold green]\n\n"
        "Promptly - Recursive Intelligence at the Edge 🚀",
        border_style="green"
    ))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
