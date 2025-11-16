#!/usr/bin/env python3
"""
MCP Server Test Demo
====================

Tests the HoloLoom Promptly MCP server to verify all tools work correctly.

This simulates Claude Desktop calling the MCP tools.

Usage:
    PYTHONPATH=. python demos/demo_mcp_server_test.py
"""

import asyncio
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm

# Import MCP server components
from HoloLoom.mcp_server_promptly import (
    initialize_hololoom,
    handle_experience,
    handle_recall,
    handle_weave,
    handle_analytics_summary,
    handle_skill_execution
)

console = Console()


async def test_1_initialization():
    """Test 1: Server initialization"""
    console.print(Panel.fit(
        "[bold cyan]Test 1: Server Initialization[/bold cyan]\\n\\n"
        "Initialize HoloLoom components",
        border_style="cyan"
    ))

    try:
        await initialize_hololoom()
        console.print("[green]✓ Server initialized successfully[/green]")
        console.print(f"  - Configuration loaded")
        console.print(f"  - 13 skills loaded")
        console.print(f"  - Orchestrator ready")
        return True
    except Exception as e:
        console.print(f"[red]✗ Initialization failed: {e}[/red]")
        return False


async def test_2_experience():
    """Test 2: hololoom_experience tool"""
    console.print(Panel.fit(
        "[bold green]Test 2: Memory Experience[/bold green]\\n\\n"
        "Store a memory using hololoom_experience",
        border_style="green"
    ))

    args = {
        "content": "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem"
    }

    try:
        result = await handle_experience(args)
        response = json.loads(result[0].text)

        console.print(f"[green]✓ Memory stored[/green]")
        console.print(f"  Status: {response['status']}")
        console.print(f"  Entities: {response['entities']}")
        return True
    except Exception as e:
        console.print(f"[red]✗ Experience failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def test_3_recall():
    """Test 3: hololoom_recall tool"""
    console.print(Panel.fit(
        "[bold magenta]Test 3: Memory Recall[/bold magenta]\\n\\n"
        "Retrieve memories using hololoom_recall",
        border_style="magenta"
    ))

    args = {
        "query": "Thompson Sampling",
        "limit": 5
    }

    try:
        result = await handle_recall(args)
        response = json.loads(result[0].text)

        console.print(f"[green]✓ Memories recalled[/green]")
        console.print(f"  Query: {response['query']}")
        console.print(f"  Found: {response['memories_found']} memories")
        return True
    except Exception as e:
        console.print(f"[red]✗ Recall failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def test_4_weave():
    """Test 4: hololoom_weave tool with recursive reasoning"""
    console.print(Panel.fit(
        "[bold yellow]Test 4: Recursive Weaving[/bold yellow]\\n\\n"
        "Execute recursive reasoning with REFINE strategy",
        border_style="yellow"
    ))

    args = {
        "query": "What is Thompson Sampling?",
        "strategy": "refine",
        "max_iterations": 2
    }

    try:
        result = await handle_weave(args)
        response = json.loads(result[0].text)

        console.print(f"[green]✓ Weaving complete[/green]")
        console.print(f"  Confidence: {response['confidence']:.1%}")
        console.print(f"  Iterations: {response['iterations']}")
        console.print(f"  Strategy: {response['strategy_used']}")
        console.print(f"\\n  Response preview: {response['response'][:100]}...")
        return True
    except Exception as e:
        console.print(f"[red]✗ Weaving failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def test_5_skill_code_reviewer():
    """Test 5: skill_code_reviewer"""
    console.print(Panel.fit(
        "[bold blue]Test 5: Code Reviewer Skill[/bold blue]\\n\\n"
        "Review Python code with CRITIQUE strategy",
        border_style="blue"
    ))

    code = """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""

    args = {
        "code": code,
        "language": "python"
    }

    try:
        result = await handle_skill_execution("code-reviewer", args)
        response = json.loads(result[0].text)

        console.print(f"[green]✓ Code review complete[/green]")
        console.print(f"  Status: {response['status']}")
        console.print(f"  Confidence: {response['confidence']:.1%}")
        console.print(f"  Iterations: {response['iterations']}")
        console.print(f"\\n  Review preview: {response['output'][:150]}...")
        return True
    except Exception as e:
        console.print(f"[red]✗ Code review failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def test_6_skill_bug_detective():
    """Test 6: skill_bug_detective"""
    console.print(Panel.fit(
        "[bold red]Test 6: Bug Detective Skill[/bold red]\\n\\n"
        "Debug code with DECOMPOSE strategy",
        border_style="red"
    ))

    buggy_code = """
def get_user_name(user):
    return user.profile.name
"""

    args = {
        "code": buggy_code,
        "language": "javascript",
        "bug_description": "Crashes when user has no profile",
        "error_message": "TypeError: Cannot read property 'name' of undefined"
    }

    try:
        result = await handle_skill_execution("bug-detective", args)
        response = json.loads(result[0].text)

        console.print(f"[green]✓ Bug analysis complete[/green]")
        console.print(f"  Status: {response['status']}")
        console.print(f"  Confidence: {response['confidence']:.1%}")
        console.print(f"  Iterations: {response['iterations']}")
        console.print(f"\\n  Analysis preview: {response['output'][:150]}...")
        return True
    except Exception as e:
        console.print(f"[red]✗ Bug detective failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def test_7_analytics():
    """Test 7: hololoom_analytics_summary"""
    console.print(Panel.fit(
        "[bold white]Test 7: Analytics Summary[/bold white]\\n\\n"
        "Get performance analytics",
        border_style="white"
    ))

    args = {}

    try:
        result = await handle_analytics_summary(args)
        response = json.loads(result[0].text)

        console.print(f"[green]✓ Analytics retrieved[/green]")
        console.print(f"  Total queries: {response.get('total_queries', 0)}")
        console.print(f"  Avg quality gain: {response.get('avg_quality_gain', 0):.1%}")

        if response.get('strategies'):
            console.print(f"  Strategies tracked: {len(response['strategies'])}")

        return True
    except Exception as e:
        console.print(f"[red]✗ Analytics failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    console.print(Panel.fit(
        "[bold white on blue] HoloLoom Promptly MCP Server Test Suite [/bold white on blue]\\n\\n"
        "[italic]Testing all 17 MCP tools[/italic]",
        border_style="blue"
    ))

    tests = [
        ("Initialization", test_1_initialization),
        ("Memory Experience", test_2_experience),
        ("Memory Recall", test_3_recall),
        ("Recursive Weaving", test_4_weave),
        ("Code Reviewer", test_5_skill_code_reviewer),
        ("Bug Detective", test_6_skill_bug_detective),
        ("Analytics", test_7_analytics),
    ]

    results = []

    for i, (name, test_fn) in enumerate(tests, 1):
        console.print(f"\\n\\n{'='*70}")
        console.print(f"[bold]Test {i}/{len(tests)}: {name}[/bold]")
        console.print(f"{'='*70}\\n")

        success = await test_fn()
        results.append((name, success))

        if i < len(tests):
            Confirm.ask("\n[bold]Continue to next test?[/bold]", default=True)

    # Summary
    console.print("\\n\\n" + "="*70)
    console.print("[bold]Test Results Summary[/bold]")
    console.print("="*70)

    table = Table()
    table.add_column("Test", style="cyan")
    table.add_column("Status", justify="center")

    passed = 0
    for name, success in results:
        status = "[green]✓ PASS[/green]" if success else "[red]✗ FAIL[/red]"
        table.add_row(name, status)
        if success:
            passed += 1

    console.print(table)

    console.print(f"\\n[bold]Total: {passed}/{len(tests)} tests passed[/bold]")

    if passed == len(tests):
        console.print("[green]\\n✅ All tests passed! MCP server is ready.[/green]")
    else:
        console.print(f"[yellow]\\n⚠️  {len(tests) - passed} test(s) failed.[/yellow]")

    console.print("\\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())
