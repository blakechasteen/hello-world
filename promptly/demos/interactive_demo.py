"""
Promptly Interactive Demo Runner

Main entry point for the interactive demo suite with rich terminal UI.
Supports both interactive menu mode and direct demo execution.

Usage:
    # Interactive menu
    python -m promptly.demos.interactive_demo

    # Run specific demo
    python -m promptly.demos.interactive_demo --demo basic
    python -m promptly.demos.interactive_demo --demo versioning
    python -m promptly.demos.interactive_demo --demo chains
    python -m promptly.demos.interactive_demo --demo judge
    python -m promptly.demos.interactive_demo --demo hololoom

    # Run all demos (automated)
    python -m promptly.demos.interactive_demo --all
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum

# Rich imports with fallback
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.box import ROUNDED
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None


class DemoType(Enum):
    """Available demo types."""
    BASIC = "basic"
    VERSIONING = "versioning"
    CHAINS = "chains"
    JUDGE = "judge"
    HOLOLOOM = "hololoom"


@dataclass
class DemoInfo:
    """Information about a demo."""
    name: str
    title: str
    description: str
    emoji: str
    module: str


# Demo registry
DEMOS: Dict[DemoType, DemoInfo] = {
    DemoType.BASIC: DemoInfo(
        name="basic",
        title="Basic Prompt Management",
        description="Create, retrieve, update prompts with variable substitution",
        emoji="📝",
        module="promptly.demos.demo_basic_prompts"
    ),
    DemoType.VERSIONING: DemoInfo(
        name="versioning",
        title="Git-like Versioning",
        description="Branches, checkouts, history, and version comparison",
        emoji="🌿",
        module="promptly.demos.demo_versioning"
    ),
    DemoType.CHAINS: DemoInfo(
        name="chains",
        title="Chains & Skills",
        description="Multi-step chains and reusable skill templates",
        emoji="⛓️",
        module="promptly.demos.demo_chains_skills"
    ),
    DemoType.JUDGE: DemoInfo(
        name="judge",
        title="LLM Judge Evaluation",
        description="12 criteria, 6 judging methods, quality scoring",
        emoji="⚖️",
        module="promptly.demos.demo_llm_judge"
    ),
    DemoType.HOLOLOOM: DemoInfo(
        name="hololoom",
        title="HoloLoom Integration",
        description="Thompson Sampling, learning insights, semantic search",
        emoji="🧠",
        module="promptly.demos.demo_hololoom"
    ),
}


class DemoRunner:
    """
    Main demo runner with rich terminal UI.

    Provides interactive menu, progress indicators, and colorful output.
    Works with or without LLM backend (mock mode).
    """

    def __init__(self, mock_mode: bool = False):
        """
        Initialize demo runner.

        Args:
            mock_mode: If True, simulate LLM operations without actual calls
        """
        self.mock_mode = mock_mode
        self.console = Console() if RICH_AVAILABLE else None
        self._sample_data_loaded = False

    def print(self, *args, **kwargs):
        """Print with rich console or fallback to print."""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            # Simple fallback
            text = " ".join(str(a) for a in args)
            print(text)

    def print_header(self):
        """Print the demo suite header."""
        if not RICH_AVAILABLE:
            print("\n" + "=" * 50)
            print("  PROMPTLY INTERACTIVE DEMO")
            print("=" * 50 + "\n")
            return

        header = Panel(
            Text.from_markup(
                "[bold cyan]🎯 Promptly Interactive Demo[/bold cyan]\n\n"
                "[dim]Explore prompt management, versioning, chains, "
                "evaluation, and HoloLoom integration[/dim]"
            ),
            box=ROUNDED,
            padding=(1, 2),
        )
        self.console.print(header)

    def print_menu(self) -> str:
        """Print interactive menu and get user choice."""
        if not RICH_AVAILABLE:
            print("\nAvailable Demos:")
            for i, (dtype, info) in enumerate(DEMOS.items(), 1):
                print(f"  [{i}] {info.title}")
            print("  [6] Quick Start Tutorial")
            print("  [7] Run All Demos (automated)")
            print("  [Q] Quit")
            return input("\nYour choice: ").strip()

        # Build menu table
        table = Table(
            show_header=False,
            box=ROUNDED,
            padding=(0, 1),
            expand=True
        )
        table.add_column("Option", style="cyan", width=6)
        table.add_column("Demo", style="white")
        table.add_column("Description", style="dim")

        for i, (dtype, info) in enumerate(DEMOS.items(), 1):
            table.add_row(
                f"[{i}]",
                f"{info.emoji} {info.title}",
                info.description
            )

        table.add_row("", "", "")  # Spacer
        table.add_row("[6]", "📚 Quick Start Tutorial", "Learn Promptly basics")
        table.add_row("[7]", "🚀 Run All Demos", "Automated walkthrough")
        table.add_row("[Q]", "❌ Quit", "Exit demo suite")

        self.console.print(table)
        self.console.print()

        return Prompt.ask(
            "[cyan]Your choice[/cyan]",
            choices=["1", "2", "3", "4", "5", "6", "7", "q", "Q"],
            default="1"
        )

    def print_demo_header(self, demo_type: DemoType):
        """Print header for a specific demo."""
        info = DEMOS[demo_type]

        if not RICH_AVAILABLE:
            print(f"\n{'=' * 50}")
            print(f"  {info.emoji} {info.title}")
            print(f"  {info.description}")
            print(f"{'=' * 50}\n")
            return

        panel = Panel(
            Text.from_markup(
                f"[bold]{info.emoji} {info.title}[/bold]\n\n"
                f"[dim]{info.description}[/dim]"
            ),
            title=f"Demo: {info.name}",
            border_style="green",
            box=ROUNDED,
            padding=(1, 2),
        )
        self.console.print(panel)

    async def run_demo(self, demo_type: DemoType) -> bool:
        """
        Run a specific demo.

        Args:
            demo_type: Which demo to run

        Returns:
            True if demo completed successfully
        """
        self.print_demo_header(demo_type)
        info = DEMOS[demo_type]

        # Ensure sample data is loaded
        if not self._sample_data_loaded:
            await self.setup_sample_data()

        try:
            # Import and run demo module
            if demo_type == DemoType.BASIC:
                from .demo_basic_prompts import run_basic_demo
                await run_basic_demo(self)
            elif demo_type == DemoType.VERSIONING:
                from .demo_versioning import run_versioning_demo
                await run_versioning_demo(self)
            elif demo_type == DemoType.CHAINS:
                from .demo_chains_skills import run_chains_demo
                await run_chains_demo(self)
            elif demo_type == DemoType.JUDGE:
                from .demo_llm_judge import run_judge_demo
                await run_judge_demo(self)
            elif demo_type == DemoType.HOLOLOOM:
                from .demo_hololoom import run_hololoom_demo
                await run_hololoom_demo(self)

            self.print_success(f"Demo '{info.name}' completed!")
            return True

        except ImportError as e:
            self.print_error(f"Demo module not found: {e}")
            return False
        except Exception as e:
            self.print_error(f"Demo error: {e}")
            return False

    async def run_tutorial(self):
        """Run quick start tutorial."""
        if not RICH_AVAILABLE:
            print("\n📚 Quick Start Tutorial")
            print("-" * 40)
            print("1. Create prompts with 'promptly add <name>'")
            print("2. Use variables: {{variable}} in prompts")
            print("3. Version with branches and commits")
            print("4. Chain prompts for complex workflows")
            print("5. Evaluate quality with LLM Judge")
            print("-" * 40)
            return

        tutorial = """
# 📚 Promptly Quick Start

## Core Concepts

1. **Prompts** - Reusable text templates with `{{variables}}`
2. **Versions** - Git-like history with SHA-256 hashes
3. **Branches** - Experiment without affecting main
4. **Skills** - Validated prompt templates with schemas
5. **Chains** - Multi-step prompt sequences
6. **Judge** - LLM-based quality evaluation

## Quick Commands

```bash
# Add a prompt
promptly add summarize "Summarize: {{text}}"

# List prompts
promptly list

# Create a branch
promptly branch experiment

# Evaluate quality
promptly eval summarize --input "Long text..."
```

## Storage Locations

- **Global**: `~/.promptly/` (shared across projects)
- **Local**: `.promptly/` (project-specific)

Local prompts override global. Auto-discovery on init.

## Next Steps

Run the individual demos to explore each feature in depth!
"""
        self.console.print(Markdown(tutorial))

        if Confirm.ask("\n[cyan]Continue to demo menu?[/cyan]", default=True):
            return

    async def run_all_demos(self, interactive: bool = False):
        """
        Run all demos in sequence.

        Args:
            interactive: If True, pause between demos
        """
        self.print("\n[bold]🚀 Running All Demos[/bold]\n")

        for demo_type in DemoType:
            await self.run_demo(demo_type)

            if interactive:
                if not RICH_AVAILABLE:
                    input("\nPress Enter to continue...")
                else:
                    if not Confirm.ask("\n[cyan]Continue to next demo?[/cyan]", default=True):
                        break

            self.print()

    async def setup_sample_data(self):
        """Load sample data for demos."""
        if self._sample_data_loaded:
            return

        self.print("[dim]Loading sample data...[/dim]")

        # Import Promptly core
        try:
            from ..core.promptly import Promptly

            # Initialize Promptly
            self.promptly = Promptly()

            # Load sample prompts
            sample_prompts = [
                ("summarize", "Summarize the following text in {{length}} sentences:\n\n{{text}}"),
                ("code-review", "Review this {{language}} code for bugs and best practices:\n\n{{code}}"),
                ("explain", "Explain {{topic}} to a {{audience}} in simple terms."),
                ("translate", "Translate the following text from {{source_lang}} to {{target_lang}}:\n\n{{text}}"),
                ("analyze", "Analyze the following data and provide insights:\n\n{{data}}"),
            ]

            for name, content in sample_prompts:
                self.promptly.add(name, content)

            self._sample_data_loaded = True
            self.print("[green]✓[/green] Sample data loaded")

        except Exception as e:
            self.print(f"[yellow]⚠ Could not load sample data: {e}[/yellow]")
            self.promptly = None

    async def interactive_loop(self):
        """Run the interactive menu loop."""
        self.print_header()

        while True:
            choice = self.print_menu()

            if choice.lower() == 'q':
                self.print("\n[cyan]Thanks for trying Promptly![/cyan] 👋\n")
                break

            try:
                choice_num = int(choice)

                if 1 <= choice_num <= 5:
                    demo_type = list(DemoType)[choice_num - 1]
                    await self.run_demo(demo_type)
                elif choice_num == 6:
                    await self.run_tutorial()
                elif choice_num == 7:
                    await self.run_all_demos(interactive=True)

            except ValueError:
                self.print("[red]Invalid choice. Please try again.[/red]")

            # Pause before returning to menu
            if RICH_AVAILABLE:
                self.console.print()
                Prompt.ask("[dim]Press Enter to continue[/dim]", default="")
            else:
                input("\nPress Enter to continue...")

    # Utility methods for demos

    def print_success(self, message: str):
        """Print success message."""
        if RICH_AVAILABLE:
            self.console.print(f"[green]✓[/green] {message}")
        else:
            print(f"✓ {message}")

    def print_error(self, message: str):
        """Print error message."""
        if RICH_AVAILABLE:
            self.console.print(f"[red]✗[/red] {message}")
        else:
            print(f"✗ {message}")

    def print_info(self, message: str):
        """Print info message."""
        if RICH_AVAILABLE:
            self.console.print(f"[blue]ℹ[/blue] {message}")
        else:
            print(f"ℹ {message}")

    def print_warning(self, message: str):
        """Print warning message."""
        if RICH_AVAILABLE:
            self.console.print(f"[yellow]⚠[/yellow] {message}")
        else:
            print(f"⚠ {message}")

    def print_prompt(self, name: str, content: str, version: str = None):
        """Print a prompt with formatting."""
        if not RICH_AVAILABLE:
            print(f"\n{name}" + (f" (v{version[:8]})" if version else ""))
            print("-" * 40)
            print(content)
            print("-" * 40)
            return

        title = f"[bold cyan]{name}[/bold cyan]"
        if version:
            title += f" [dim](v{version[:8]})[/dim]"

        panel = Panel(
            content,
            title=title,
            border_style="cyan",
            box=ROUNDED,
        )
        self.console.print(panel)

    def print_table(self, title: str, headers: List[str], rows: List[List[str]]):
        """Print a formatted table."""
        if not RICH_AVAILABLE:
            print(f"\n{title}")
            print("-" * 60)
            print(" | ".join(headers))
            print("-" * 60)
            for row in rows:
                print(" | ".join(str(cell) for cell in row))
            print("-" * 60)
            return

        table = Table(title=title, box=ROUNDED)
        for header in headers:
            table.add_column(header, style="cyan")
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        self.console.print(table)

    def print_json(self, data: Dict[str, Any], title: str = None):
        """Print JSON data with formatting."""
        import json
        formatted = json.dumps(data, indent=2)

        if not RICH_AVAILABLE:
            if title:
                print(f"\n{title}")
            print(formatted)
            return

        from rich.syntax import Syntax
        syntax = Syntax(formatted, "json", theme="monokai", line_numbers=False)
        if title:
            panel = Panel(syntax, title=title, border_style="blue", box=ROUNDED)
            self.console.print(panel)
        else:
            self.console.print(syntax)

    def show_progress(self, description: str):
        """Create a progress context for long operations."""
        if not RICH_AVAILABLE:
            print(f"{description}...")
            return DummyProgress()

        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        )

    def prompt_choice(self, message: str, choices: List[str], default: str = None) -> str:
        """Prompt user for a choice."""
        if not RICH_AVAILABLE:
            print(f"{message} ({'/'.join(choices)})")
            return input("> ").strip() or default

        return Prompt.ask(
            f"[cyan]{message}[/cyan]",
            choices=choices,
            default=default
        )

    def prompt_confirm(self, message: str, default: bool = True) -> bool:
        """Prompt user for confirmation."""
        if not RICH_AVAILABLE:
            response = input(f"{message} (y/n): ").strip().lower()
            return response in ('y', 'yes', '')

        return Confirm.ask(f"[cyan]{message}[/cyan]", default=default)

    def prompt_input(self, message: str, default: str = None) -> str:
        """Prompt user for text input."""
        if not RICH_AVAILABLE:
            prompt_str = f"{message}"
            if default:
                prompt_str += f" [{default}]"
            prompt_str += ": "
            response = input(prompt_str).strip()
            return response or default

        return Prompt.ask(f"[cyan]{message}[/cyan]", default=default)


class DummyProgress:
    """Dummy progress context for non-rich mode."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def add_task(self, description, total=None):
        return 0
    def update(self, task_id, advance=1):
        pass


# Convenience functions

async def run_interactive_demo(mock_mode: bool = False):
    """Run the interactive demo suite."""
    runner = DemoRunner(mock_mode=mock_mode)
    await runner.interactive_loop()


async def run_demo(demo_name: str, mock_mode: bool = False) -> bool:
    """
    Run a specific demo by name.

    Args:
        demo_name: Name of demo (basic, versioning, chains, judge, hololoom)
        mock_mode: If True, simulate LLM operations

    Returns:
        True if demo completed successfully
    """
    runner = DemoRunner(mock_mode=mock_mode)

    try:
        demo_type = DemoType(demo_name.lower())
        return await runner.run_demo(demo_type)
    except ValueError:
        runner.print_error(f"Unknown demo: {demo_name}")
        runner.print_info(f"Available demos: {', '.join(d.value for d in DemoType)}")
        return False


async def setup_sample_data():
    """Load sample data for demos."""
    runner = DemoRunner()
    await runner.setup_sample_data()
    return runner.promptly


# CLI entry point

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Promptly Interactive Demo Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m promptly.demos.interactive_demo           # Interactive menu
  python -m promptly.demos.interactive_demo --demo basic
  python -m promptly.demos.interactive_demo --demo judge
  python -m promptly.demos.interactive_demo --all     # Run all demos
"""
    )
    parser.add_argument(
        "--demo", "-d",
        choices=["basic", "versioning", "chains", "judge", "hololoom"],
        help="Run a specific demo"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all demos in sequence"
    )
    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        help="Run in mock mode (no LLM calls)"
    )

    args = parser.parse_args()

    if args.demo:
        asyncio.run(run_demo(args.demo, mock_mode=args.mock))
    elif args.all:
        runner = DemoRunner(mock_mode=args.mock)
        asyncio.run(runner.run_all_demos(interactive=False))
    else:
        asyncio.run(run_interactive_demo(mock_mode=args.mock))


if __name__ == "__main__":
    main()
