#!/usr/bin/env python3
"""
Ollama + Awareness Demo
========================

Demonstrates ACTUAL LLM integration with compositional awareness.

This is the **real thing** - not templates, but actual Ollama-generated responses
guided by compositional awareness context.

Requirements:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull llama3.2:3b
    3. Install Python package: pip install ollama

Features demonstrated:
- Ollama LLM integration with awareness context
- Template fallback if Ollama unavailable
- Comparison: LLM vs template responses
- Meta-awareness + actual LLM reasoning

Usage:
    python demos/demo_ollama_awareness.py
"""

import asyncio
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.awareness import (
    CompositionalAwarenessLayer,
    DualStreamGenerator,
    MetaAwarenessLayer,
    LLM_AVAILABLE
)
from rich.console import Console
from rich.panel import Panel
from rich import box


async def demo_ollama_integration():
    """Demo actual Ollama LLM integration"""
    console = Console()

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/]")
    console.print("[bold magenta]  Ollama + Awareness: Real LLM Integration[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/]\n")

    # Check LLM availability
    if not LLM_AVAILABLE:
        console.print("[red]❌ LLM integration not available[/]")
        console.print("[yellow]Install with: pip install ollama[/]")
        console.print("[dim]Falling back to template mode...[/]\n")
        use_llm = False
    else:
        from HoloLoom.awareness import OllamaLLM

        # Try to initialize Ollama
        try:
            llm = OllamaLLM(model="llama3.2:3b")
            if llm.is_available():
                console.print("[green]✓ Ollama available![/]")
                console.print(f"[dim]Model: {llm.model}[/]\n")
                use_llm = True
            else:
                console.print("[yellow]⚠ Ollama server not running[/]")
                console.print("[dim]Start with: ollama serve[/]")
                console.print("[dim]Falling back to template mode...[/]\n")
                use_llm = False
                llm = None
        except Exception as e:
            console.print(f"[red]❌ Ollama error: {e}[/]")
            console.print("[dim]Falling back to template mode...[/]\n")
            use_llm = False
            llm = None

    # Initialize awareness layers
    awareness = CompositionalAwarenessLayer()

    # Initialize generator (with or without LLM)
    if use_llm:
        console.print("[bold green]Using ACTUAL Ollama LLM for generation[/]\n")
        generator = DualStreamGenerator(awareness, llm_generator=llm)
    else:
        console.print("[bold yellow]Using template-based generation (no LLM)[/]\n")
        generator = DualStreamGenerator(awareness, llm_generator=None)

    # Test queries
    test_queries = [
        "What is Thompson Sampling?",
        "How does compositional awareness work?",
        "What is quantum entanglement?"
    ]

    for i, query in enumerate(test_queries, 1):
        console.print(f"\n[bold]═══ Query {i}/{len(test_queries)} ═══[/]")
        console.print(f"[cyan]Query:[/] {query}\n")

        # Generate response
        console.print("[yellow]Generating...[/]")
        dual_stream = await generator.generate(query)

        # Display external response
        panel = Panel(
            dual_stream.external_stream,
            title=f"[bold green]Response ({'LLM' if use_llm else 'Template'})[/]",
            border_style="green",
            box=box.ROUNDED
        )
        console.print(panel)

        # Show awareness metrics
        ctx = dual_stream.awareness_context
        console.print(f"[dim]Confidence: {1.0 - ctx.confidence.uncertainty_level:.2f} | "
                     f"Domain: {ctx.patterns.domain} | "
                     f"Cache: {ctx.confidence.query_cache_status}[/]")

        # Ask if user wants to see internal reasoning
        if i < len(test_queries):
            console.print("[dim]Press Enter for next query...[/]")
            input()

    console.print("\n[bold green]Demo complete![/]\n")

    if use_llm:
        console.print("[bold]Key Differences (LLM vs Template):[/]")
        console.print("  ✓ LLM responses are natural and fluent")
        console.print("  ✓ LLM adapts to awareness context (tone, hedging)")
        console.print("  ✓ LLM provides actual knowledge, not just meta-info")
        console.print("  ✓ Template shows analysis but not real answers")
    else:
        console.print("[bold]To use actual LLM:[/]")
        console.print("  1. Install Ollama: https://ollama.ai")
        console.print("  2. Run: ollama pull llama3.2:3b")
        console.print("  3. Run: pip install ollama")
        console.print("  4. Start server: ollama serve")
        console.print("  5. Run this demo again!")


async def demo_comparison():
    """Side-by-side comparison: LLM vs Template"""
    console = Console()

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/]")
    console.print("[bold magenta]  LLM vs Template Comparison[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/]\n")

    # Check Ollama availability
    if not LLM_AVAILABLE:
        console.print("[yellow]Ollama not available - skipping comparison[/]")
        return

    from HoloLoom.awareness import OllamaLLM

    try:
        llm = OllamaLLM(model="llama3.2:3b")
        if not llm.is_available():
            console.print("[yellow]Ollama not available - skipping comparison[/]")
            return
    except:
        console.print("[yellow]Ollama error - skipping comparison[/]")
        return

    # Initialize both generators
    awareness = CompositionalAwarenessLayer()
    llm_gen = DualStreamGenerator(awareness, llm_generator=llm)
    template_gen = DualStreamGenerator(awareness, llm_generator=None)

    query = "What is Thompson Sampling used for in reinforcement learning?"

    console.print(f"[bold]Query:[/] {query}\n")

    # Generate template response
    console.print("[yellow]Generating template response...[/]")
    template_stream = await template_gen.generate(query)

    # Generate LLM response
    console.print("[yellow]Generating Ollama response...[/]")
    llm_stream = await llm_gen.generate(query)

    # Show side-by-side
    from rich.columns import Columns

    template_panel = Panel(
        template_stream.external_stream,
        title="[bold yellow]Template Response[/]",
        border_style="yellow",
        box=box.ROUNDED
    )

    llm_panel = Panel(
        llm_stream.external_stream,
        title="[bold green]Ollama Response[/]",
        border_style="green",
        box=box.ROUNDED
    )

    console.print(Columns([template_panel, llm_panel]))

    console.print("\n[bold]Notice the difference:[/]")
    console.print("  • Template: Shows meta-info about confidence and domain")
    console.print("  • Ollama: Provides actual knowledge about Thompson Sampling")
    console.print("  • Both: Guided by the same compositional awareness context")


def main():
    """Run demo"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        asyncio.run(demo_comparison())
    else:
        asyncio.run(demo_ollama_integration())


if __name__ == "__main__":
    main()
