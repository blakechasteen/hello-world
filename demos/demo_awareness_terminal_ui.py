#!/usr/bin/env python3
"""
Awareness Terminal UI Demo
===========================

Demonstrates the compositional awareness layer integrated into the rich terminal UI.

Features demonstrated:
1. Compositional awareness (structural, pattern, confidence analysis)
2. Dual-stream generation (internal reasoning + external response)
3. Meta-awareness (recursive self-reflection, epistemic humility)
4. Interactive 'awareness' command to explore context

Usage:
    python demos/demo_awareness_terminal_ui.py

Commands during demo:
    - Just type queries naturally
    - Type 'awareness' to see full compositional awareness context
    - Type 'history' to see conversation history
    - Type 'stats' for session statistics
    - Type 'quit' to exit
"""

import asyncio
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.awareness import (
    CompositionalAwarenessLayer,
    DualStreamGenerator,
    MetaAwarenessLayer
)
from HoloLoom.terminal_ui import TerminalUI
from rich.console import Console


async def demo_interactive():
    """
    Interactive demo - you can type queries and explore awareness.
    """
    console = Console()

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/]")
    console.print("[bold magenta]  Awareness Terminal UI - Interactive Demo[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/]\n")

    console.print("[dim]This demo shows compositional awareness integrated into HoloLoom's terminal UI.[/]\n")

    # Initialize terminal UI with awareness
    console.print("[yellow]Initializing awareness layers...[/]")
    ui = TerminalUI(orchestrator=None, enable_awareness=True)

    console.print("\n[green]✓ Awareness enabled![/]")
    console.print("\n[bold]Try these example queries:[/]")
    console.print("  1. [cyan]'What is a red ball?'[/] (familiar pattern)")
    console.print("  2. [cyan]'What is quantum entanglement?'[/] (novel pattern)")
    console.print("  3. [cyan]'How do I train a model?'[/] (ambiguous - what kind of model?)")
    console.print("\n[dim]After each query, type 'awareness' to see the full context![/]\n")

    # Run interactive session
    await ui.interactive_session()


async def demo_automated():
    """
    Automated demo showing different query types and awareness contexts.
    """
    console = Console()

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/]")
    console.print("[bold magenta]  Awareness Terminal UI - Automated Demo[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/]\n")

    # Initialize terminal UI with awareness
    console.print("[yellow]Initializing awareness layers...[/]")
    ui = TerminalUI(orchestrator=None, enable_awareness=True)
    console.print("[green]✓ Awareness enabled![/]\n")

    # Test queries with different characteristics
    test_queries = [
        {
            "query": "What is a red ball?",
            "description": "Familiar pattern (physical objects)",
            "expected_confidence": "HIGH",
        },
        {
            "query": "What is quantum superposition in recursive meta-learning?",
            "description": "Novel pattern (ambiguous technical terms)",
            "expected_confidence": "LOW",
        },
        {
            "query": "How does Thompson Sampling work?",
            "description": "Technical question (familiar if seen before)",
            "expected_confidence": "MEDIUM",
        },
    ]

    for i, test in enumerate(test_queries, 1):
        console.print(f"\n[bold]═══ Test {i}/3: {test['description']} ═══[/]")
        console.print(f"[dim]Query: \"{test['query']}\"[/]")
        console.print(f"[dim]Expected: {test['expected_confidence']} confidence[/]\n")

        # Generate awareness-guided response
        dual_stream = await ui.weave_with_awareness(
            test['query'],
            pattern=None,
            show_awareness_live=False
        )

        # Show awareness context
        console.print("\n[bold cyan]→ Compositional Awareness Context:[/]")
        ui.show_awareness_context(dual_stream.awareness_context)

        # Ask if user wants to see dual-stream
        console.print("[dim]Press Enter to continue to next query...[/]")
        input()

    # Show session summary
    console.print("\n\n[bold green]═══ Demo Complete ═══[/]\n")
    ui.show_stats()

    console.print("\n[bold]Key Takeaways:[/]")
    console.print("  ✓ Compositional awareness analyzes query structure and patterns")
    console.print("  ✓ Confidence signals guide tone and response strategy")
    console.print("  ✓ Meta-awareness provides recursive self-reflection")
    console.print("  ✓ 'awareness' command gives full introspection anytime")
    console.print("\n[dim]Try the interactive demo for hands-on exploration![/]\n")


async def demo_meta_awareness():
    """
    Focused demo showing meta-awareness (recursive self-reflection).
    """
    console = Console()

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/]")
    console.print("[bold magenta]  Meta-Awareness: Recursive Self-Reflection Demo[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/]\n")

    # Initialize awareness layers directly
    awareness = CompositionalAwarenessLayer()
    generator = DualStreamGenerator(awareness)
    meta = MetaAwarenessLayer(awareness)

    query = "What is quantum meta-recursive emergence?"

    console.print(f"[bold]Query:[/] \"{query}\"")
    console.print("[dim](Intentionally ambiguous to trigger high uncertainty)[/]\n")

    # Generate response
    console.print("[yellow]Generating awareness-guided response...[/]")
    dual_stream = await generator.generate(query, show_internal=True)

    # Show external response
    console.print("\n[bold green]External Response:[/]")
    console.print(f"  {dual_stream.external_stream}\n")

    # Perform meta-reflection
    console.print("[yellow]Performing recursive self-reflection...[/]")
    reflection = await meta.recursive_self_reflection(
        query=query,
        response=dual_stream.external_stream,
        awareness_context=dual_stream.awareness_context
    )

    # Show meta-awareness introspection
    ui = TerminalUI(orchestrator=None, enable_awareness=False)
    ui.show_meta_reflection(reflection)

    console.print("\n[bold]Meta-Awareness Insights:[/]")
    console.print(f"  • Uncertainty decomposition identifies [yellow]{reflection.uncertainty_decomposition.dominant_type.value}[/] as dominant")
    console.print(f"  • Meta-confidence: [cyan]{reflection.meta_confidence.meta_confidence:.2f}[/] (confidence about confidence)")
    console.print(f"  • Epistemic humility: [cyan]{reflection.epistemic_humility:.2f}[/] (appropriately humble)")
    console.print(f"  • Knowledge gaps: [yellow]{len(reflection.detected_gaps)}[/] detected")
    console.print(f"  • Adversarial probes: [yellow]{len(reflection.adversarial_probes)}[/] self-tests generated")
    console.print("\n[dim]This is AI examining its own reasoning process![/]\n")


def main():
    """Run demo based on command-line argument"""
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        console = Console()
        console.print("\n[bold]Choose demo mode:[/]")
        console.print("  1. [cyan]interactive[/] - Type your own queries")
        console.print("  2. [cyan]automated[/] - See predefined examples")
        console.print("  3. [cyan]meta[/] - Deep dive into meta-awareness")
        choice = input("\nChoice (1/2/3): ").strip()

        mode_map = {"1": "interactive", "2": "automated", "3": "meta"}
        mode = mode_map.get(choice, "interactive")

    if mode == "automated":
        asyncio.run(demo_automated())
    elif mode == "meta":
        asyncio.run(demo_meta_awareness())
    else:
        asyncio.run(demo_interactive())


if __name__ == "__main__":
    main()
