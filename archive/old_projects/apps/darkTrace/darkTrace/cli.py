"""
darkTrace Command-Line Interface
=================================
CLI for semantic analysis and LLM reverse engineering.

Usage:
    darktrace analyze input.txt --config narrative
    darktrace fingerprint model_output.jsonl --output fingerprint.json
    darktrace monitor --config dialogue --live
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, List

try:
    import click
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.syntax import Syntax
except ImportError:
    click = None
    Console = None
    print(
        "ERROR: CLI dependencies not installed.\n"
        "Install with: pip install 'darktrace-llm[cli]'",
        file=sys.stderr
    )
    sys.exit(1)

from darkTrace import DarkTraceConfig
from darkTrace.api import create_api, DarkTraceAPI


console = Console()


def get_config(config_name: str) -> DarkTraceConfig:
    """Get configuration by name."""
    configs = {
        "bare": DarkTraceConfig.bare,
        "fast": DarkTraceConfig.fast,
        "fused": DarkTraceConfig.fused,
        "narrative": DarkTraceConfig.narrative,
        "dialogue": DarkTraceConfig.dialogue,
        "technical": DarkTraceConfig.technical,
    }

    factory = configs.get(config_name.lower())
    if not factory:
        console.print(f"[red]Unknown config: {config_name}[/red]")
        console.print(f"Available: {', '.join(configs.keys())}")
        sys.exit(1)

    return factory()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    darkTrace - Semantic Reverse Engineering of LLMs

    Analyze LLM outputs, generate fingerprints, and predict trajectories
    using 244D semantic analysis.
    """
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--config', '-c', default='fast', help='Config preset (bare/fast/fused/narrative/dialogue/technical)')
@click.option('--output', '-o', type=click.Path(), help='Output file (JSON)')
@click.option('--predict/--no-predict', default=True, help='Enable trajectory prediction')
@click.option('--patterns/--no-patterns', default=True, help='Enable pattern detection')
@click.option('--attractors/--no-attractors', default=True, help='Enable attractor detection')
def analyze(
    input_file: str,
    config: str,
    output: Optional[str],
    predict: bool,
    patterns: bool,
    attractors: bool,
):
    """
    Analyze text file and display semantic insights.

    Example:
        darktrace analyze odyssey.txt --config narrative
    """
    asyncio.run(_analyze(input_file, config, output, predict, patterns, attractors))


async def _analyze(
    input_file: str,
    config_name: str,
    output: Optional[str],
    predict: bool,
    patterns: bool,
    attractors: bool,
):
    """Async implementation of analyze."""
    # Read input
    text = Path(input_file).read_text(encoding='utf-8')

    # Create API
    config = get_config(config_name)
    api = create_api(config_name)

    console.print(Panel(
        f"[bold cyan]darkTrace Semantic Analysis[/bold cyan]\n\n"
        f"Input: {input_file}\n"
        f"Config: {config_name}\n"
        f"Length: {len(text)} characters",
        title="Configuration"
    ))

    # Analyze with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing...", total=None)

        async with api:
            result = await api.analyze_text(
                text,
                predict=predict,
                detect_patterns=patterns,
                detect_attractors=attractors,
            )

        progress.update(task, completed=True)

    # Display results
    _display_analysis_result(result)

    # Save to file if requested
    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))
        console.print(f"\n[green]✓[/green] Results saved to {output}")


def _display_analysis_result(result):
    """Display analysis result in terminal."""
    # Dominant dimensions
    table = Table(title="Dominant Dimensions", show_header=True)
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", justify="right", style="green")

    for dim in result.dominant_dimensions[:10]:
        score = result.dimension_scores.get(dim, 0.0)
        table.add_row(dim, f"{score:.3f}")

    console.print(table)

    # Trajectory metrics
    console.print("\n[bold]Trajectory Metrics[/bold]")
    console.print(f"  Velocity: {result.velocity_magnitude:.3f}")
    console.print(f"  Acceleration: {result.acceleration_magnitude:.3f}")
    if result.curvature is not None:
        console.print(f"  Curvature: {result.curvature:.3f}")

    # Ethics
    if result.ethical_valence is not None:
        console.print(f"\n[bold]Ethical Analysis[/bold]")
        console.print(f"  Valence: {result.ethical_valence:.3f}")

    # Performance
    console.print(f"\n[dim]Analysis completed in {result.duration_ms:.1f}ms[/dim]")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--config', '-c', default='fast', help='Config preset')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output fingerprint file (JSON)')
@click.option('--model-name', '-m', help='Model name/identifier')
def fingerprint(
    input_file: str,
    config: str,
    output: str,
    model_name: Optional[str],
):
    """
    Generate semantic fingerprint from text samples.

    Input file should be JSONL with one sample per line.

    Example:
        darktrace fingerprint model_output.jsonl -o fingerprint.json
    """
    asyncio.run(_fingerprint(input_file, config, output, model_name))


async def _fingerprint(
    input_file: str,
    config_name: str,
    output: str,
    model_name: Optional[str],
):
    """Async implementation of fingerprint."""
    # Read samples
    texts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Support different formats
                text = data.get('text') or data.get('output') or data.get('content') or str(data)
                texts.append(text)

    console.print(Panel(
        f"[bold cyan]darkTrace Fingerprint Generation[/bold cyan]\n\n"
        f"Input: {input_file}\n"
        f"Config: {config_name}\n"
        f"Samples: {len(texts)}\n"
        f"Model: {model_name or 'Unknown'}",
        title="Configuration"
    ))

    # Generate fingerprint
    api = create_api(config_name)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing {len(texts)} samples...", total=None)

        async with api:
            result = await api.generate_fingerprint(texts, model_name=model_name)

        progress.update(task, completed=True)

    # Save fingerprint
    Path(output).write_text(json.dumps(result.to_dict(), indent=2))
    console.print(f"\n[green]✓[/green] Fingerprint saved to {output}")

    # Display summary
    console.print("\n[bold]Fingerprint Summary[/bold]")
    console.print(f"  Dimensions: {result.fingerprint_dimensions}")
    console.print(f"  Samples: {result.num_samples}")
    console.print(f"  Patterns: {len(result.signature_patterns)}")
    console.print(f"  Attractors: {len(result.attractor_locations)}")


@cli.command()
@click.option('--config', '-c', default='dialogue', help='Config preset')
@click.option('--live/--no-live', default=False, help='Enable live monitoring')
def monitor(config: str, live: bool):
    """
    Monitor LLM outputs in real-time.

    Example:
        darktrace monitor --config dialogue --live
    """
    asyncio.run(_monitor(config, live))


async def _monitor(config_name: str, live: bool):
    """Async implementation of monitor."""
    console.print(Panel(
        f"[bold cyan]darkTrace Real-Time Monitor[/bold cyan]\n\n"
        f"Config: {config_name}\n"
        f"Live mode: {'enabled' if live else 'disabled'}",
        title="Monitor"
    ))

    api = create_api(config_name)

    async with api:
        console.print("\n[yellow]Waiting for input (Ctrl+C to exit)...[/yellow]\n")

        try:
            while True:
                # Read from stdin
                console.print("[dim]Enter text:[/dim]", end=" ")
                text = await asyncio.get_event_loop().run_in_executor(None, input)

                if not text.strip():
                    continue

                # Analyze
                result = await api.analyze_text(text, predict=True)

                # Display
                console.print(f"\n[cyan]→ Dominant:[/cyan] {', '.join(result.dominant_dimensions[:3])}")
                console.print(f"[cyan]→ Velocity:[/cyan] {result.velocity_magnitude:.3f}")
                console.print()

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Monitoring stopped.[/yellow]")


@cli.command()
@click.option('--config', '-c', default='fast', help='Config preset')
def status(config: str):
    """
    Display system status and metrics.

    Example:
        darktrace status
    """
    asyncio.run(_status(config))


async def _status(config_name: str):
    """Async implementation of status."""
    api = create_api(config_name)

    async with api:
        status = await api.get_status()

    # Display status
    console.print(Panel(
        f"[bold]Status:[/bold] {status.status}\n"
        f"[bold]Uptime:[/bold] {status.uptime_seconds:.1f}s\n"
        f"[bold]Analyses:[/bold] {status.total_analyses}\n"
        f"[bold]Avg Time:[/bold] {status.avg_analysis_time_ms:.1f}ms",
        title="darkTrace System Status",
        border_style="green" if status.status == "healthy" else "red"
    ))

    # Active layers
    console.print("\n[bold]Active Layers[/bold]")
    for layer in status.active_layers:
        console.print(f"  • {layer}")

    # Component status
    console.print("\n[bold]Components[/bold]")
    console.print(f"  Observer: {'✓' if status.observer_ready else '✗'}")
    console.print(f"  Analyzer: {'✓' if status.analyzer_ready else '✗'}")


def main():
    """Entry point for CLI."""
    try:
        cli()
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
