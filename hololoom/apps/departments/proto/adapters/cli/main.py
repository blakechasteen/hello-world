"""Proto CLI adapter - Command-line interface for Proto code agent.

Commands:
    ask <question>    - Ask Proto a question
    repl              - Start interactive REPL session
    review <file>     - Review a code file
    explain <file>    - Explain a code file
    --version         - Show version
    --help            - Show help

Usage:
    proto ask "Explain how this function works"
    proto review path/to/file.py
    proto explain path/to/complex_module.py
    proto repl

Implementation: 2025-12-02
Status: Production ready
"""

import asyncio
import sys
from typing import Optional
from pathlib import Path

# Try typer first (prettier), fall back to click
try:
    import typer
    USE_TYPER = True
    app = typer.Typer(
        name="proto",
        help="Proto - Your relaxed code assistant",
        add_completion=False,
        pretty_exceptions_enable=False,
    )
except ImportError:
    import click
    USE_TYPER = False
    app = None


VERSION = "0.1.0"


class ProtoCLI:
    """Proto CLI adapter class for programmatic usage."""

    def __init__(self):
        """Initialize CLI adapter."""
        self.engine = None

    async def init_engine(self):
        """Initialize Proto engine lazily."""
        if self.engine is None:
            from hololoom.apps.departments.proto.core import ProtoEngine, ProtoConfig

            config = ProtoConfig.default()
            self.engine = ProtoEngine(config)
            await self.engine.startup()
        return self.engine

    async def process_query(
        self, query: str, context_file: Optional[str] = None
    ) -> str:
        """Process a query through Proto engine.

        Args:
            query: The query text
            context_file: Optional path to context file

        Returns:
            Response text from Proto
        """
        from hololoom.apps.departments.proto.domain import CodeContext

        engine = await self.init_engine()

        try:
            # Build code context if file provided
            code_context = None
            if context_file:
                path = Path(context_file)
                if path.exists():
                    content = path.read_text()
                    code_context = CodeContext(
                        file_path=str(path),
                        content=content,
                        language=path.suffix.lstrip(".") or "text",
                    )

            # Process query through engine
            response = await engine.process(query, code_context)
            return response.content

        except Exception as e:
            raise RuntimeError(f"Failed to process query: {e}")

    async def shutdown(self):
        """Shutdown engine and cleanup resources."""
        if self.engine is not None:
            await self.engine.shutdown()
            self.engine = None


def run_async(coro):
    """Run async coroutine in sync context."""
    return asyncio.run(coro)


# Global CLI instance
_cli_instance = ProtoCLI()


async def start_repl():
    """Start interactive REPL session.

    Uses the full-featured REPL from repl.py module with Rich formatting,
    command history, and context management.
    """
    from .repl import start_repl as repl_main

    await repl_main()


if USE_TYPER:
    # ============================================================================
    # Typer CLI (prettier output, modern CLI UX)
    # ============================================================================

    @app.command()
    def ask(
        question: str = typer.Argument(
            ..., help="Question to ask Proto"
        ),
        file: Optional[str] = typer.Option(
            None, "--file", "-f", help="Context file path"
        ),
    ):
        """Ask Proto a question about code or concepts."""
        try:
            cli = _cli_instance
            result = run_async(cli.process_query(question, file))
            typer.echo(f"\n{result}\n")
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
        finally:
            run_async(cli.shutdown())

    @app.command()
    def repl():
        """Start interactive REPL session."""
        try:
            run_async(start_repl())
        except ImportError as e:
            typer.echo(
                "Error: REPL dependencies may be missing.\n"
                f"Details: {e}",
                err=True,
            )
            raise typer.Exit(1)

    @app.command()
    def review(
        file: str = typer.Argument(..., help="File to review"),
    ):
        """Review a code file for quality and best practices."""
        if not Path(file).exists():
            typer.echo(f"Error: File not found: {file}", err=True)
            raise typer.Exit(1)

        try:
            cli = _cli_instance
            result = run_async(cli.process_query("Review this code", file))
            typer.echo(f"\n{result}\n")
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
        finally:
            run_async(cli.shutdown())

    @app.command()
    def explain(
        file: str = typer.Argument(..., help="File to explain"),
    ):
        """Explain a code file or function."""
        if not Path(file).exists():
            typer.echo(f"Error: File not found: {file}", err=True)
            raise typer.Exit(1)

        try:
            cli = _cli_instance
            result = run_async(cli.process_query("Explain this code", file))
            typer.echo(f"\n{result}\n")
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
        finally:
            run_async(cli.shutdown())

    @app.callback(invoke_without_command=True)
    def main(
        version: bool = typer.Option(
            False, "--version", "-v", help="Show version"
        ),
        ctx: typer.Context = typer.Context,
    ):
        """Proto - Your relaxed code assistant.

        Ask questions, review code, and get explanations with Proto.
        """
        if version:
            typer.echo(f"Proto v{VERSION}")
            raise typer.Exit()

else:
    # ============================================================================
    # Click CLI (fallback for minimal dependencies)
    # ============================================================================

    @click.group()
    @click.version_option(VERSION, prog_name="proto")
    def app():
        """Proto - Your relaxed code assistant."""
        pass

    @app.command()
    @click.argument("question")
    @click.option(
        "--file", "-f", default=None, help="Context file path"
    )
    def ask(question: str, file: Optional[str]):
        """Ask Proto a question about code or concepts."""
        try:
            cli = _cli_instance
            result = run_async(cli.process_query(question, file))
            click.echo(f"\n{result}\n")
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            run_async(cli.shutdown())

    @app.command()
    def repl():
        """Start interactive REPL session."""
        try:
            run_async(start_repl())
        except ImportError as e:
            click.echo(
                "Error: REPL dependencies may be missing.\n"
                f"Details: {e}",
                err=True,
            )
            sys.exit(1)

    @app.command()
    @click.argument("file")
    def review(file: str):
        """Review a code file for quality and best practices."""
        if not Path(file).exists():
            click.echo(f"Error: File not found: {file}", err=True)
            sys.exit(1)

        try:
            cli = _cli_instance
            result = run_async(cli.process_query("Review this code", file))
            click.echo(f"\n{result}\n")
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            run_async(cli.shutdown())

    @app.command()
    @click.argument("file")
    def explain(file: str):
        """Explain a code file or function."""
        if not Path(file).exists():
            click.echo(f"Error: File not found: {file}", err=True)
            sys.exit(1)

        try:
            cli = _cli_instance
            result = run_async(cli.process_query("Explain this code", file))
            click.echo(f"\n{result}\n")
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            run_async(cli.shutdown())


def cli():
    """Entry point for Proto CLI."""
    if USE_TYPER:
        app()
    else:
        app()


if __name__ == "__main__":
    cli()
