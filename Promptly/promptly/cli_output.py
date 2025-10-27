#!/usr/bin/env python3
"""
Rich CLI Output Module
======================
Beautiful terminal output with colors, tables, progress bars, and formatting.

Features:
- Colored output (success, error, warning, info)
- Tables with auto-sizing
- Progress bars for loops
- Syntax highlighting
- Panels and boxes
- Tree views for hierarchies
"""

from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from rich.markdown import Markdown
from rich import box
from rich.text import Text
import time
import sys

# Global console instance with proper encoding
console = Console(force_terminal=True, legacy_windows=False)

# Use ASCII-safe symbols for Windows
if sys.platform == 'win32':
    SYMBOL_SUCCESS = "[+]"
    SYMBOL_ERROR = "[!]"
    SYMBOL_WARNING = "[*]"
    SYMBOL_INFO = "[i]"
else:
    SYMBOL_SUCCESS = "✓"
    SYMBOL_ERROR = "✗"
    SYMBOL_WARNING = "⚠"
    SYMBOL_INFO = "ℹ"


# ============================================================================
# Basic Output Functions
# ============================================================================

def success(message: str):
    """Print success message in green"""
    console.print(f"[bold green]{SYMBOL_SUCCESS}[/bold green] {message}")


def error(message: str):
    """Print error message in red"""
    console.print(f"[bold red]{SYMBOL_ERROR}[/bold red] {message}")


def warning(message: str):
    """Print warning message in yellow"""
    console.print(f"[bold yellow]{SYMBOL_WARNING}[/bold yellow] {message}")


def info(message: str):
    """Print info message in blue"""
    console.print(f"[bold blue]{SYMBOL_INFO}[/bold blue] {message}")


def heading(message: str, style: str = "bold cyan"):
    """Print heading"""
    console.print(f"\n[{style}]{message}[/{style}]")


def separator(char: str = "─", width: Optional[int] = None):
    """Print separator line"""
    if width is None:
        width = console.width
    # Use ASCII-safe character on Windows
    if sys.platform == 'win32':
        char = "-"
    console.print(char * width, style="dim")


# ============================================================================
# Tables
# ============================================================================

def create_table(title: str, headers: List[str], rows: List[List[str]], **kwargs) -> Table:
    """
    Create a rich table.

    Args:
        title: Table title
        headers: Column headers
        rows: List of rows (each row is list of strings)
        **kwargs: Additional Table arguments

    Returns:
        Table object
    """
    table = Table(
        title=title,
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        **kwargs
    )

    # Add columns
    for header in headers:
        table.add_column(header)

    # Add rows
    for row in rows:
        table.add_row(*row)

    return table


def print_table(title: str, headers: List[str], rows: List[List[str]], **kwargs):
    """Print a table directly"""
    table = create_table(title, headers, rows, **kwargs)
    console.print(table)


def print_dict_table(data: Dict[str, Any], title: str = "Data"):
    """Print dictionary as a two-column table"""
    table = Table(title=title, box=box.SIMPLE, show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")

    for key, value in data.items():
        table.add_row(str(key), str(value))

    console.print(table)


# ============================================================================
# Progress Bars
# ============================================================================

class LoopProgress:
    """Context manager for loop progress tracking"""

    def __init__(self, task_description: str, total: int):
        self.task_description = task_description
        self.total = total
        self.progress = None
        self.task_id = None

    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        )
        self.progress.__enter__()
        self.task_id = self.progress.add_task(self.task_description, total=self.total)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.__exit__(exc_type, exc_val, exc_tb)

    def update(self, advance: int = 1):
        """Update progress"""
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, advance=advance)


def with_spinner(description: str):
    """Decorator for showing spinner during function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with console.status(f"[bold blue]{description}...") as status:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


# ============================================================================
# Panels and Boxes
# ============================================================================

def print_panel(content: str, title: str = "", style: str = "cyan", **kwargs):
    """Print content in a panel/box"""
    panel = Panel(
        content,
        title=title,
        border_style=style,
        box=box.ROUNDED,
        **kwargs
    )
    console.print(panel)


def print_code(code: str, language: str = "python", theme: str = "monokai"):
    """Print syntax-highlighted code"""
    syntax = Syntax(code, language, theme=theme, line_numbers=True)
    console.print(syntax)


def print_markdown(markdown_text: str):
    """Print markdown-formatted text"""
    md = Markdown(markdown_text)
    console.print(md)


# ============================================================================
# Trees
# ============================================================================

def create_tree(label: str, style: str = "bold cyan") -> Tree:
    """Create a tree structure"""
    return Tree(f"[{style}]{label}[/{style}]")


def print_loop_result_tree(result: Any):
    """
    Print recursive loop result as a tree.

    Args:
        result: LoopResult object
    """
    tree = create_tree("Loop Execution Result")

    tree.add(f"[cyan]Iterations:[/cyan] {result.iterations}")
    tree.add(f"[cyan]Stop Reason:[/cyan] {result.stop_reason}")
    tree.add(f"[cyan]Final Output:[/cyan] {len(result.final_output)} chars")

    if result.improvement_history:
        history_node = tree.add("[cyan]Quality History[/cyan]")
        for i, quality in enumerate(result.improvement_history, 1):
            color = "green" if quality > 0.8 else "yellow" if quality > 0.6 else "red"
            history_node.add(f"[{color}]Iteration {i}: {quality:.2f}[/{color}]")

    console.print(tree)


# ============================================================================
# Prompt-Specific Formatters
# ============================================================================

def print_prompt_summary(prompt_data: Dict[str, Any]):
    """
    Print a formatted prompt summary.

    Args:
        prompt_data: Dict with prompt information
    """
    # Create title
    title = f"Prompt: {prompt_data.get('name', 'Unknown')}"

    # Create content
    lines = []
    lines.append(f"[bold]ID:[/bold] {prompt_data.get('id', 'N/A')}")
    lines.append(f"[bold]Version:[/bold] {prompt_data.get('version', 1)}")

    if 'tags' in prompt_data and prompt_data['tags']:
        tags = ", ".join(f"[yellow]{tag}[/yellow]" for tag in prompt_data['tags'])
        lines.append(f"[bold]Tags:[/bold] {tags}")

    if 'usage_count' in prompt_data:
        lines.append(f"[bold]Usage:[/bold] {prompt_data['usage_count']} times")

    if 'avg_quality' in prompt_data:
        quality = prompt_data['avg_quality']
        color = "green" if quality > 0.8 else "yellow" if quality > 0.6 else "red"
        lines.append(f"[bold]Quality:[/bold] [{color}]{quality:.2f}[/{color}]")

    content = "\n".join(lines)

    print_panel(content, title=title, style="cyan")


def print_analytics_summary(analytics: Dict[str, Any]):
    """
    Print analytics summary with rich formatting.

    Args:
        analytics: Analytics dict from PromptAnalytics.get_summary()
    """
    heading("Analytics Summary", "bold magenta")

    # Overall stats table
    stats_table = Table(box=box.SIMPLE, show_header=False)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white", justify="right")

    stats_table.add_row("Total Executions", str(analytics.get('total_executions', 0)))
    stats_table.add_row("Unique Prompts", str(analytics.get('unique_prompts', 0)))
    stats_table.add_row("Success Rate", f"{analytics.get('success_rate', 0):.1%}")
    stats_table.add_row("Avg Execution Time", f"{analytics.get('avg_execution_time', 0):.2f}s")

    if analytics.get('avg_quality') is not None:
        quality = analytics['avg_quality']
        color = "green" if quality > 0.8 else "yellow" if quality > 0.6 else "red"
        stats_table.add_row("Avg Quality", f"[{color}]{quality:.2f}[/{color}]")

    if analytics.get('total_cost', 0) > 0:
        stats_table.add_row("Total Cost", f"${analytics['total_cost']:.2f}")

    if analytics.get('total_tokens', 0) > 0:
        stats_table.add_row("Total Tokens", f"{analytics['total_tokens']:,}")

    console.print(stats_table)


def print_search_results(results: List[Dict[str, Any]], query: str):
    """
    Print search results in a formatted table.

    Args:
        results: List of search result dicts
        query: The search query
    """
    heading(f"Search Results for '{query}'", "bold cyan")

    if not results:
        warning("No results found")
        return

    # Create results table
    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED
    )

    table.add_column("#", style="dim", width=3)
    table.add_column("Name", style="cyan")
    table.add_column("Tags", style="yellow")
    table.add_column("Relevance", justify="right")

    for i, result in enumerate(results, 1):
        name = result.get('context', {}).get('name', result.get('text', 'Unknown'))
        tags = ", ".join(result.get('context', {}).get('tags', [])[:3])
        relevance = result.get('relevance', 0.0)

        # Color code relevance
        if relevance > 0.8:
            rel_color = "green"
        elif relevance > 0.6:
            rel_color = "yellow"
        else:
            rel_color = "red"

        table.add_row(
            str(i),
            name[:50],
            tags[:30] if tags else "-",
            f"[{rel_color}]{relevance:.2f}[/{rel_color}]"
        )

    console.print(table)


def print_loop_execution(loop_type: str, iterations: int, final_quality: float):
    """
    Print loop execution summary.

    Args:
        loop_type: Type of loop (REFINE, CRITIQUE, etc.)
        iterations: Number of iterations
        final_quality: Final quality score
    """
    # Choose symbol based on loop type (ASCII-safe for Windows)
    if sys.platform == 'win32':
        symbol_map = {
            'REFINE': '[R]',
            'CRITIQUE': '[C]',
            'DECOMPOSE': '[D]',
            'VERIFY': '[V]',
            'EXPLORE': '[E]',
            'HOFSTADTER': '[H]'
        }
        symbol = symbol_map.get(loop_type, '[L]')
        quality_symbols = {
            'high': '[*]',
            'medium': '[~]',
            'low': '[!]'
        }
    else:
        symbol_map = {
            'REFINE': '🔧',
            'CRITIQUE': '🔍',
            'DECOMPOSE': '🧩',
            'VERIFY': '✓',
            'EXPLORE': '🌐',
            'HOFSTADTER': '♾️'
        }
        symbol = symbol_map.get(loop_type, '🔄')
        quality_symbols = {
            'high': '✨',
            'medium': '⚡',
            'low': '⚠️'
        }

    # Color code quality
    if final_quality > 0.8:
        quality_color = "green"
        quality_symbol = quality_symbols['high']
    elif final_quality > 0.6:
        quality_color = "yellow"
        quality_symbol = quality_symbols['medium']
    else:
        quality_color = "red"
        quality_symbol = quality_symbols['low']

    console.print(
        f"{symbol} [bold]{loop_type}[/bold] loop completed: "
        f"{iterations} iterations, "
        f"quality: [{quality_color}]{final_quality:.2f}[/{quality_color}] {quality_symbol}"
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test all formatting functions
    heading("Rich CLI Output Demo")

    # Basic messages
    success("Operation completed successfully!")
    error("Something went wrong!")
    warning("This is a warning")
    info("Just FYI")

    separator()

    # Table
    print_table(
        "Top Prompts",
        ["Name", "Quality", "Usage"],
        [
            ["SQL Optimizer", "0.87", "42"],
            ["Code Reviewer", "0.92", "67"],
            ["Bug Detective", "0.85", "89"]
        ]
    )

    separator()

    # Dict table
    print_dict_table({
        "Total Executions": 340,
        "Unique Prompts": 15,
        "Success Rate": "94.2%",
        "Avg Quality": "0.88"
    }, title="Analytics Summary")

    separator()

    # Panel
    print_panel(
        "This is a [bold]panel[/bold] with [cyan]colored[/cyan] content!",
        title="Example Panel",
        style="green"
    )

    separator()

    # Progress bar demo
    heading("Progress Bar Demo")
    with LoopProgress("Processing iterations", total=10) as progress:
        for i in range(10):
            time.sleep(0.1)
            progress.update()

    success("Demo complete!")
