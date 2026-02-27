"""
HoloLoom Lite - Rich Terminal UI

A beautiful terminal interface using the Rich library.

Requirements:
    pip install rich

Usage:
    python -m HoloLoom.lite terminal

Features:
    - Colored output with syntax highlighting
    - Progress spinners during processing
    - Formatted tables for memories
    - Conversation panels with threading
    - Real-time metrics display

Date: December 2025
"""

import asyncio
from typing import List, Tuple, Optional


def start_terminal(**kwargs) -> None:
    """Start the rich terminal UI."""
    asyncio.run(_terminal_main(**kwargs))


async def _terminal_main(**kwargs) -> None:
    """Main terminal loop with Rich formatting."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.prompt import Prompt
        from rich.spinner import Spinner
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.text import Text
        from rich.layout import Layout
        from rich import box
    except ImportError:
        print("Rich terminal requires 'rich' package.")
        print("Install with: pip install rich")
        return

    from hololoom.lite import HoloLoomLite

    console = Console()
    history: List[Tuple[str, str, float]] = []

    # Header
    console.print()
    console.print(Panel.fit(
        "[bold blue]HoloLoom Lite[/bold blue]\n"
        "[dim]Simplified AI with memory[/dim]",
        border_style="blue",
        padding=(1, 4)
    ))
    console.print()

    # Commands help
    console.print("[dim]Commands: /help, /history, /memories, /metrics, /clear, /quit[/dim]")
    console.print("[dim]Modes: /reason <mode> <query> (direct, verify, research, plan_execute)[/dim]")
    console.print()

    console.print("[yellow]Initializing...[/yellow]")

    async with HoloLoomLite() as loom:
        console.print("[green]Ready![/green]\n")

        while True:
            try:
                # Get input with styled prompt
                user_input = Prompt.ask("[bold cyan]>>>[/bold cyan]").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command = user_input[1:].lower().split()[0]

                    if command in ("quit", "exit", "q"):
                        console.print("\n[bold green]Goodbye![/bold green]")
                        break

                    elif command == "help":
                        _rich_help(console)

                    elif command == "history":
                        _rich_history(console, history)

                    elif command == "memories":
                        await _rich_memories(console, loom)

                    elif command == "metrics":
                        _rich_metrics(console, loom)

                    elif command == "clear":
                        history.clear()
                        console.print("[yellow]History cleared.[/yellow]")

                    elif command == "experience":
                        content = user_input[12:].strip()
                        if content:
                            with console.status("[bold green]Storing memory...", spinner="dots"):
                                mem = await loom.experience(content)
                            console.print(f"[green]Stored:[/green] {mem.id}")
                        else:
                            console.print("[yellow]Usage: /experience <content>[/yellow]")

                    elif command == "reason":
                        parts = user_input[8:].strip().split(" ", 1)
                        if len(parts) >= 2:
                            mode, query = parts
                            if mode in ("direct", "verify", "research", "plan_execute"):
                                with console.status(f"[bold green]Reasoning ({mode})...", spinner="dots"):
                                    result = await loom.reason(query, mode=mode)

                                console.print()
                                console.print(Panel(
                                    result.text,
                                    title=f"[bold blue]{mode.upper()} Response[/bold blue]",
                                    border_style="blue"
                                ))
                                console.print(f"[dim]Confidence: {result.confidence:.2%}[/dim]")
                                console.print()

                                history.append((query, result.text, result.confidence))
                            else:
                                console.print("[yellow]Mode must be: direct, verify, research, or plan_execute[/yellow]")
                        else:
                            console.print("[yellow]Usage: /reason <mode> <query>[/yellow]")

                    else:
                        console.print(f"[yellow]Unknown command: /{command}[/yellow]")

                    continue

                # Regular query
                console.print()

                # Process with spinner
                with console.status("[bold green]Thinking...", spinner="dots"):
                    await loom.experience(user_input, context={"type": "query"})
                    result = await loom.reason(user_input, mode="direct")

                # Display response in a panel
                console.print(Panel(
                    result.text,
                    title="[bold blue]Response[/bold blue]",
                    border_style="blue"
                ))

                # Show confidence
                if result.confidence >= 0.8:
                    conf_color = "green"
                elif result.confidence >= 0.5:
                    conf_color = "yellow"
                else:
                    conf_color = "red"

                console.print(f"[dim]Confidence: [{conf_color}]{result.confidence:.2%}[/{conf_color}][/dim]")
                console.print()

                history.append((user_input, result.text, result.confidence))

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type /quit to exit.[/yellow]")

            except EOFError:
                console.print("\n[bold green]Goodbye![/bold green]")
                break

            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {e}\n")


def _rich_help(console):
    """Print help with Rich formatting."""
    from rich.table import Table
    from rich import box

    table = Table(title="Commands", box=box.ROUNDED)
    table.add_column("Command", style="cyan")
    table.add_column("Description")

    commands = [
        ("/help", "Show this help"),
        ("/history", "Show conversation history"),
        ("/memories", "Show stored memories"),
        ("/metrics", "Show system metrics"),
        ("/clear", "Clear history"),
        ("/experience <text>", "Store explicit memory"),
        ("/reason <mode> <query>", "Agentic reasoning"),
        ("/quit", "Exit"),
    ]

    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Reasoning modes: direct, verify, research, plan_execute[/dim]")
    console.print()


def _rich_history(console, history: List[Tuple[str, str, float]]):
    """Print history with Rich formatting."""
    from rich.table import Table
    from rich import box

    if not history:
        console.print("[yellow]No history yet.[/yellow]")
        return

    table = Table(title=f"History ({len(history)} turns)", box=box.ROUNDED)
    table.add_column("#", style="dim", width=4)
    table.add_column("Query", style="cyan")
    table.add_column("Response")
    table.add_column("Conf", style="green", width=6)

    for i, (query, response, conf) in enumerate(history[-10:], 1):
        q_short = query[:40] + "..." if len(query) > 40 else query
        r_short = response[:40] + "..." if len(response) > 40 else response
        table.add_row(str(i), q_short, r_short, f"{conf:.0%}")

    console.print()
    console.print(table)
    console.print()


async def _rich_memories(console, loom):
    """Print memories with Rich formatting."""
    from rich.table import Table
    from rich import box

    metrics = loom.get_metrics()

    console.print()
    console.print(f"[bold]Total Memories:[/bold] {metrics['n_memories']}")
    console.print(f"[bold]Connections:[/bold] {metrics['n_connections']}")
    console.print()

    if metrics['n_memories'] > 0:
        memories = await loom.recall("", limit=10)
        if memories:
            table = Table(title="Recent Memories", box=box.ROUNDED)
            table.add_column("ID", style="dim", width=12)
            table.add_column("Content")

            for mem in memories[:10]:
                text = mem.text[:60] + "..." if len(mem.text) > 60 else mem.text
                table.add_row(mem.id[:10] + "...", text)

            console.print(table)
    console.print()


def _rich_metrics(console, loom):
    """Print metrics with Rich formatting."""
    from rich.panel import Panel

    metrics = loom.get_metrics()

    content = f"""
Memories:    {metrics['n_memories']}
Connections: {metrics['n_connections']}
Initialized: {metrics['initialized']}
"""

    console.print()
    console.print(Panel(content.strip(), title="[bold blue]Metrics[/bold blue]"))
    console.print()


if __name__ == "__main__":
    start_terminal()
