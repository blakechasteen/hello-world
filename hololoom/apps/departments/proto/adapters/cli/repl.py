from __future__ import annotations
"""
Proto Interactive REPL
======================
Interactive read-eval-print loop for Proto.

Features:
- Command history
- Multi-line input
- Special commands (/help, /quit, /clear, /context)
- Session memory
- Colored output (optional, via Rich)
"""

import asyncio
from pathlib import Path

# Try Rich for pretty output
try:
    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False


BANNER = """
╔═══════════════════════════════════════════════════════════╗
║  Proto - Your relaxed code assistant                       ║
║  Type /help for commands, /quit to exit                    ║
╚═══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Commands:
  /help     - Show this help
  /quit     - Exit REPL
  /clear    - Clear screen
  /context  - Set context file
  /history  - Show recent history

Paste code blocks or ask questions directly.
Multi-line input: end with blank line.
"""


def print_output(text: str, style: str = None):
    """Print output, with Rich formatting if available."""
    if RICH_AVAILABLE and console:
        if style == "markdown":
            console.print(Markdown(text))
        elif style:
            console.print(text, style=style)
        else:
            console.print(text)
    else:
        print(text)


def clear_screen():
    """Clear terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


async def start_repl():
    """Start the interactive REPL."""
    from hololoom.apps.departments.proto.core import ProtoConfig, ProtoEngine
    from hololoom.apps.departments.proto.domain import CodeContext, ProtoSession

    # Initialize
    config = ProtoConfig.default()
    config.mode = "interactive"
    engine = ProtoEngine(config)
    await engine.startup()

    session = ProtoSession()
    context_file: Path | None = None
    history = []

    try:
        print_output(BANNER, style="bold cyan")

        while True:
            try:
                # Get input
                prompt = "proto> " if not context_file else f"proto [{context_file.name}]> "

                if RICH_AVAILABLE and console:
                    user_input = console.input(f"[bold green]{prompt}[/]")
                else:
                    user_input = input(prompt)

                user_input = user_input.strip()

                # Skip empty
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    cmd = user_input.lower()

                    if cmd == '/quit' or cmd == '/exit':
                        print_output("Goodbye!", style="bold cyan")
                        break

                    elif cmd == '/help':
                        print_output(HELP_TEXT)
                        continue

                    elif cmd == '/clear':
                        clear_screen()
                        print_output(BANNER, style="bold cyan")
                        continue

                    elif cmd.startswith('/context'):
                        parts = user_input.split(maxsplit=1)
                        if len(parts) > 1:
                            path = Path(parts[1])
                            if path.exists():
                                context_file = path
                                print_output(f"Context set: {path}", style="green")
                            else:
                                print_output(f"File not found: {path}", style="red")
                        else:
                            context_file = None
                            print_output("Context cleared", style="yellow")
                        continue

                    elif cmd == '/history':
                        if history:
                            for i, h in enumerate(history[-10:], 1):
                                print_output(f"{i}. {h[:50]}...")
                        else:
                            print_output("No history yet")
                        continue

                    else:
                        print_output(f"Unknown command: {cmd}", style="red")
                        continue

                # Add to history
                history.append(user_input)

                # Build context
                code_context = None
                if context_file and context_file.exists():
                    code_context = CodeContext(
                        file_path=str(context_file),
                        content=context_file.read_text(),
                        language=context_file.suffix.lstrip('.') or 'text'
                    )

                # Process query
                print_output("Thinking...", style="dim")

                response = await engine.process(user_input, code_context)

                # Display response
                print_output("")
                if RICH_AVAILABLE:
                    print_output(response.content, style="markdown")
                else:
                    print(response.content)
                print_output("")

                # Add to session
                session.add_turn("user", user_input)
                session.add_turn("proto", response.content)

            except KeyboardInterrupt:
                print_output("\nUse /quit to exit", style="yellow")
                continue

            except EOFError:
                break

    finally:
        await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(start_repl())
