#!/usr/bin/env python3
"""
Skill Agent System Demo
=======================

Demonstrates HoloLoom's professional skill agents integrated with
Promptly's recursive reasoning.

Shows:
1. Loading skill registry
2. Listing available skills
3. Executing skills (code-reviewer, bug-detective, test-generator)
4. Recursive reasoning strategies
5. Quality-driven refinement
6. Complete provenance tracking

Usage:
    PYTHONPATH=. python demos/demo_skill_agents.py
"""

import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from hololoom.config import Config
from hololoom.agentic.skill_agents import (
    SkillRegistry,
    SkillExecutor,
    execute_skill,
    list_available_skills
)

console = Console()


async def demo_1_list_skills():
    """Demo 1: List all available skills"""
    console.print(Panel.fit(
        "[bold cyan]Demo 1: Available Skills[/bold cyan]\\n\\n"
        "Load and list all professional skill templates",
        border_style="cyan"
    ))

    # List skills by category
    skills_by_category = await list_available_skills()

    # Create table
    table = Table(title="HoloLoom Professional Skills (13 Total)")
    table.add_column("Category", style="cyan")
    table.add_column("Skills", style="white")

    for category, skills in sorted(skills_by_category.items()):
        skills_str = ", ".join(skills)
        table.add_row(category, skills_str)

    console.print(table)
    console.print(f"\\n[bold]Total Skills Loaded:[/bold] {sum(len(s) for s in skills_by_category.values())}")


async def demo_2_code_review():
    """Demo 2: Execute code-reviewer skill"""
    console.print(Panel.fit(
        "[bold green]Demo 2: Code Review[/bold green]\\n\\n"
        "Review code using the code-reviewer skill with CRITIQUE strategy",
        border_style="green"
    ))

    # Sample code to review
    code = """
def process_user_data(data):
    results = []
    for item in data:
        if item["age"] > 18:
            results.append(item["name"].upper())
    return results
"""

    console.print("[bold]Code to Review:[/bold]")
    console.print(Syntax(code, "python", theme="monokai", line_numbers=True))

    # Execute skill
    console.print("\\n[dim]Executing code-reviewer skill...[/dim]")

    result = await execute_skill(
        skill_name="code-reviewer",
        parameters={
            "code": code,
            "language": "python",
            "filename": "utils.py",
            "focus_areas": "all"
        },
        config=Config.fast(),
        enable_analytics=True
    )

    # Display results
    console.print(f"\\n[bold]Results:[/bold]")
    console.print(f"  Confidence: {result.confidence:.1%}")
    console.print(f"  Iterations: {result.iterations}")
    console.print(f"  Strategy: {result.strategy_used}")
    console.print(f"  Time: {result.execution_time_ms:.1f}ms")

    console.print(f"\\n[bold]Review Output:[/bold]")
    console.print(Panel(result.output, border_style="green"))


async def demo_3_bug_detective():
    """Demo 3: Execute bug-detective skill"""
    console.print(Panel.fit(
        "[bold magenta]Demo 3: Bug Detective[/bold magenta]\\n\\n"
        "Debug code using DECOMPOSE strategy for systematic root cause analysis",
        border_style="magenta"
    ))

    # Buggy code
    code = """
def get_user_profile(user_id):
    user = database.query(user_id)
    return user.profile.name
"""

    console.print("[bold]Buggy Code:[/bold]")
    console.print(Syntax(code, "python", theme="monokai", line_numbers=True))

    # Execute bug-detective
    console.print("\\n[dim]Executing bug-detective skill...[/dim]")

    result = await execute_skill(
        skill_name="bug-detective",
        parameters={
            "code": code,
            "language": "python",
            "bug_description": "Function crashes when user has no profile",
            "error_message": "AttributeError: 'NoneType' object has no attribute 'name'",
            "expected_behavior": "Return user's name or a default value",
            "actual_behavior": "Crashes with AttributeError"
        },
        config=Config.fast()
    )

    # Display results
    console.print(f"\\n[bold]Debug Results:[/bold]")
    console.print(f"  Confidence: {result.confidence:.1%}")
    console.print(f"  Iterations: {result.iterations}")
    console.print(f"  Strategy: {result.strategy_used}")

    console.print(f"\\n[bold]Debug Analysis:[/bold]")
    console.print(Panel(result.output, border_style="magenta"))


async def demo_4_test_generator():
    """Demo 4: Execute test-generator skill"""
    console.print(Panel.fit(
        "[bold yellow]Demo 4: Test Generator[/bold yellow]\\n\\n"
        "Generate comprehensive tests using EXPLORE strategy",
        border_style="yellow"
    ))

    # Code to test
    code = """
def calculate_discount(price, discount_percent):
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    return price * (1 - discount_percent / 100)
"""

    console.print("[bold]Code to Test:[/bold]")
    console.print(Syntax(code, "python", theme="monokai", line_numbers=True))

    # Execute test-generator
    console.print("\\n[dim]Executing test-generator skill...[/dim]")

    result = await execute_skill(
        skill_name="test-generator",
        parameters={
            "code": code,
            "language": "python",
            "framework": "pytest",
            "happy_path": True,
            "edge_cases": True,
            "error_handling": True
        },
        config=Config.fast()
    )

    # Display results
    console.print(f"\\n[bold]Test Generation Results:[/bold]")
    console.print(f"  Confidence: {result.confidence:.1%}")
    console.print(f"  Iterations: {result.iterations}")
    console.print(f"  Strategy: {result.strategy_used}")

    console.print(f"\\n[bold]Generated Tests:[/bold]")
    console.print(Panel(result.output, border_style="yellow"))


async def demo_5_strategy_comparison():
    """Demo 5: Compare different reasoning strategies"""
    console.print(Panel.fit(
        "[bold red]Demo 5: Strategy Comparison[/bold red]\\n\\n"
        "Execute same skill with different recursive reasoning strategies",
        border_style="red"
    ))

    code = "def foo(x): return x * 2"

    strategies = ["refine", "critique", "decompose"]
    results = []

    for strategy in strategies:
        console.print(f"\\n[dim]Testing strategy: {strategy}...[/dim]")

        result = await execute_skill(
            skill_name="code-reviewer",
            parameters={"code": code, "language": "python"},
            config=Config.fast(),
            override_strategy=strategy,
            override_iterations=2
        )

        results.append({
            "strategy": strategy,
            "confidence": result.confidence,
            "iterations": result.iterations,
            "time_ms": result.execution_time_ms
        })

    # Comparison table
    table = Table(title="Strategy Performance Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Confidence", justify="right")
    table.add_column("Iterations", justify="right")
    table.add_column("Time (ms)", justify="right")

    for r in results:
        table.add_row(
            r["strategy"],
            f"{r['confidence']:.1%}",
            str(r["iterations"]),
            f"{r['time_ms']:.1f}"
        )

    console.print("\\n")
    console.print(table)


async def demo_6_provenance():
    """Demo 6: Show complete reasoning provenance"""
    console.print(Panel.fit(
        "[bold blue]Demo 6: Reasoning Provenance[/bold blue]\\n\\n"
        "View complete thought process via ReasoningJournal",
        border_style="blue"
    ))

    code = "SELECT * FROM users WHERE active = TRUE"

    result = await execute_skill(
        skill_name="sql-optimizer",
        parameters={
            "query": code,
            "database_type": "postgresql"
        },
        config=Config.fast(),
        override_iterations=3
    )

    console.print(f"\\n[bold]Optimization Results:[/bold]")
    console.print(f"  Iterations: {result.iterations}")
    console.print(f"  Final Confidence: {result.confidence:.1%}")

    # Show reasoning journal if available
    if result.metadata.get("reasoning_journal"):
        console.print(f"\\n[bold]Reasoning Journal:[/bold]")
        console.print(Panel(
            result.metadata["reasoning_journal"],
            border_style="blue",
            title="Complete Thought Process"
        ))

    console.print(f"\\n[bold]Optimized Query:[/bold]")
    console.print(Panel(result.output, border_style="blue"))


async def main():
    """Run all demos"""
    console.print(Panel.fit(
        "[bold white on blue] HoloLoom Skill Agent System [/bold white on blue]\\n\\n"
        "[italic]13 Professional Skills × Recursive Reasoning[/italic]",
        border_style="blue"
    ))

    demos = [
        ("List Skills", demo_1_list_skills),
        ("Code Review", demo_2_code_review),
        ("Bug Detective", demo_3_bug_detective),
        ("Test Generator", demo_4_test_generator),
        ("Strategy Comparison", demo_5_strategy_comparison),
        ("Reasoning Provenance", demo_6_provenance),
    ]

    for i, (name, demo_fn) in enumerate(demos, 1):
        console.print(f"\\n\\n{'='*70}")
        console.print(f"[bold]Demo {i}/{len(demos)}: {name}[/bold]")
        console.print(f"{'='*70}\\n")

        try:
            await demo_fn()
        except Exception as e:
            console.print(f"[red]Error in {name}: {e}[/red]")
            import traceback
            traceback.print_exc()

        if i < len(demos):
            console.print("\\n[dim]Press Enter to continue...[/dim]")
            input()

    console.print("\\n\\n" + "="*70)
    console.print("[bold green]✅ All demos complete![/bold green]")
    console.print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
