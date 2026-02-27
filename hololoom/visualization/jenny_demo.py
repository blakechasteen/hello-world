#!/usr/bin/env python3
"""
Jenny Demo - Interactive Showcase
==================================
A world-class demonstration of Jenny Generative UI Runtime.

"Disposable pixels, durable decisions."

This demo showcases:
1. Zero-config panel generation
2. Lifecycle state machine in action
3. Complete provenance tracking
4. Real-time streaming updates
5. Action handling with feedback
6. Multi-target rendering (HTML + Terminal)

Run:
    python -m HoloLoom.visualization.jenny_demo

Or:
    PYTHONPATH=. python HoloLoom/visualization/jenny_demo.py
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# Add parent to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from HoloLoom.visualization.jenny_runtime import (
    JennyRuntime,
    JennyConfig,
    JennyPanel,
    jenny_session,
)
from HoloLoom.visualization.jenny_spec import (
    LifecycleStage,
    BindingMode,
    PanelTypeJenny,
)
from HoloLoom.visualization.jenny_actions import ActionStatus
from HoloLoom.visualization.jenny_streaming import StreamUpdate


# ============================================================================
# Demo Configuration
# ============================================================================

# ANSI colors for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.ENDC}\n")


def print_subheader(text: str) -> None:
    """Print a subsection header."""
    print(f"\n{Colors.CYAN}─── {text} ───{Colors.ENDC}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_panel_card(panel: JennyPanel) -> None:
    """Print a panel as a nice card."""
    lifecycle_colors = {
        LifecycleStage.NASCENT: Colors.YELLOW,
        LifecycleStage.STABLE: Colors.GREEN,
        LifecycleStage.DISSOLVING: Colors.RED,
        LifecycleStage.ARCHIVED: Colors.DIM,
        LifecycleStage.SYSTEM: Colors.CYAN,
    }

    color = lifecycle_colors.get(panel.lifecycle, Colors.ENDC)

    print(f"┌{'─' * 58}┐")
    print(f"│ {Colors.BOLD}{panel.title[:54]:<54}{Colors.ENDC} │")
    print(f"├{'─' * 58}┤")
    print(f"│ ID: {panel.id[:48]:<48}   │")
    print(f"│ Type: {panel.panel_type.value:<50} │")
    print(f"│ Lifecycle: {color}{panel.lifecycle.value:<46}{Colors.ENDC} │")
    print(f"│ Actions: {len(panel.actions):<48} │")
    print(f"└{'─' * 58}┘")


# ============================================================================
# Demo Scenarios
# ============================================================================

async def demo_basic_flow(jenny: JennyRuntime) -> None:
    """Demo 1: Basic question → panel → pin flow."""
    print_header("Demo 1: Basic Flow - Question to Pinned Panel")

    print_info("Asking Jenny: 'What is Thompson Sampling?'")
    panel = await jenny.ask("What is Thompson Sampling?")

    print_success("Panel generated!")
    print_panel_card(panel)

    print_subheader("Panel Content Preview")
    # Show first 200 chars of HTML
    html_preview = panel.html[:200].replace('\n', ' ')
    print(f"{Colors.DIM}{html_preview}...{Colors.ENDC}")

    print_subheader("Terminal Rendering")
    print(panel.terminal[:500])

    print_subheader("Pinning the Panel")
    print_info(f"Panel is {panel.lifecycle.value} - pinning to make it STABLE...")

    result = await jenny.act(panel, "pin")

    if result.status == ActionStatus.SUCCESS:
        print_success(f"Panel pinned! {result.previous_lifecycle.value} → {result.new_lifecycle.value}")
    else:
        print_warning(f"Pin failed: {result.error}")

    # Refresh panel to see new state
    updated = await jenny.get_panel(panel.id)
    print_panel_card(updated)


async def demo_lifecycle_states(jenny: JennyRuntime) -> None:
    """Demo 2: Lifecycle state machine."""
    print_header("Demo 2: Lifecycle State Machine")

    print_info("Creating multiple panels to demonstrate lifecycle...")

    queries = [
        "Explain Python decorators",
        "What is a neural network?",
        "How does garbage collection work?",
    ]

    panels: List[JennyPanel] = []
    for query in queries:
        panel = await jenny.ask(query)
        panels.append(panel)
        print_success(f"Created: {panel.title[:40]}... ({panel.lifecycle.value})")

    print_subheader("All Panels Start as NASCENT")
    for p in panels:
        print(f"  • {p.id[:8]}... : {Colors.YELLOW}{p.lifecycle.value}{Colors.ENDC}")

    print_subheader("Pin First Panel → STABLE")
    await jenny.act(panels[0], "pin")
    panels[0] = await jenny.get_panel(panels[0].id)
    print_success(f"{panels[0].id[:8]}... is now {Colors.GREEN}{panels[0].lifecycle.value}{Colors.ENDC}")

    print_subheader("Dismiss Second Panel → DISSOLVING")
    await jenny.act(panels[1], "dismiss")
    panels[1] = await jenny.get_panel(panels[1].id)
    if panels[1]:
        print_success(f"{panels[1].id[:8]}... is now {Colors.RED}{panels[1].lifecycle.value}{Colors.ENDC}")
    else:
        print_success("Panel dissolved and cleaned up!")

    print_subheader("Final State Summary")
    all_panels = await jenny.list_panels()
    for p in all_panels:
        lifecycle_color = {
            LifecycleStage.NASCENT: Colors.YELLOW,
            LifecycleStage.STABLE: Colors.GREEN,
            LifecycleStage.DISSOLVING: Colors.RED,
        }.get(p.lifecycle, Colors.ENDC)

        print(f"  • {p.title[:40]:<40} : {lifecycle_color}{p.lifecycle.value:<10}{Colors.ENDC}")


async def demo_provenance(jenny: JennyRuntime) -> None:
    """Demo 3: Complete provenance tracking."""
    print_header("Demo 3: Complete Provenance - Every Decision Logged")

    print_info("Performing several operations...")

    # Create panel
    panel = await jenny.ask("What is recursion?")
    print_success("Panel created")

    # Pin it
    await jenny.act(panel, "pin")
    print_success("Panel pinned")

    # Show WHY
    await jenny.act(panel, "why")
    print_success("'Why this UI?' requested")

    # Copy content
    await jenny.act(panel, "copy")
    print_success("Content copied")

    print_subheader("Complete Audit Trail")
    history = jenny.history(limit=10)

    print(f"\n{Colors.BOLD}Last {len(history)} operations:{Colors.ENDC}\n")

    for entry in history:
        event_type = entry.get("event_type", "unknown")
        timestamp = entry.get("timestamp", "")

        if isinstance(timestamp, datetime):
            timestamp = timestamp.strftime("%H:%M:%S.%f")[:-3]
        elif isinstance(timestamp, str) and "T" in timestamp:
            timestamp = timestamp.split("T")[1][:12]

        spec_id = entry.get("spec_id", "")[:8]

        # Color by event type
        color = Colors.BLUE
        if "creation" in event_type:
            color = Colors.GREEN
        elif "transition" in event_type:
            color = Colors.YELLOW
        elif "action" in event_type:
            color = Colors.CYAN

        print(f"  {Colors.DIM}{timestamp}{Colors.ENDC} {color}[{event_type:<20}]{Colors.ENDC} spec:{spec_id}...")

    print_subheader("Action History")
    actions = jenny.action_history(limit=5)

    for action in actions:
        handler = action.get("handler", "unknown")
        status = action.get("status", "unknown")
        status_color = Colors.GREEN if status == "success" else Colors.RED

        print(f"  • {handler:<15} : {status_color}{status}{Colors.ENDC}")


async def demo_why_panel(jenny: JennyRuntime) -> None:
    """Demo 4: The 'Why this UI?' meta-panel."""
    print_header("Demo 4: 'Why This UI?' - Breaking the Strange Loop")

    print_info("Creating a panel and asking 'why'...")

    panel = await jenny.ask("Compare Python vs JavaScript")
    print_success(f"Created: {panel.panel_type.value} panel")

    print_subheader("Requesting Explanation")
    result = await jenny.act(panel, "why")

    if result.status == ActionStatus.SUCCESS:
        why_data = result.data.get("why_panel_data", {})

        print(f"\n{Colors.BOLD}Why this panel was generated:{Colors.ENDC}\n")

        print(f"  Original panel type: {Colors.CYAN}{why_data.get('panel_type', 'unknown')}{Colors.ENDC}")

        print(f"\n  {Colors.BOLD}Reasoning:{Colors.ENDC}")
        for reason in why_data.get("reasoning", []):
            print(f"    • {reason}")

        print(f"\n  {Colors.BOLD}Alternatives considered:{Colors.ENDC}")
        for alt in why_data.get("alternatives_considered", []):
            print(f"    • {alt}")

        print(f"\n  {Colors.BOLD}Provenance:{Colors.ENDC}")
        prov = why_data.get("provenance", {})
        print(f"    Spacetime: {prov.get('spacetime_id', 'unknown')}")
        print(f"    Compiler:  {prov.get('compiler_version', 'unknown')}")


async def demo_panel_types(jenny: JennyRuntime) -> None:
    """Demo 5: Different panel types."""
    print_header("Demo 5: Intelligent Panel Type Selection")

    queries_and_types = [
        ("What is the capital of France?", "Factual → text"),
        ("Show me Python list comprehension syntax", "Code → code"),
        ("List the benefits of exercise", "List → list"),
        ("Compare cats vs dogs", "Comparison → comparison"),
        ("Explain how photosynthesis works", "Process → graph"),
    ]

    print_info("Asking different types of questions...\n")

    for query, expected in queries_and_types:
        panel = await jenny.ask(query)

        # Truncate query for display
        short_query = query[:40] + "..." if len(query) > 40 else query

        print(f"  Q: {Colors.DIM}{short_query:<45}{Colors.ENDC}")
        print(f"     → {Colors.CYAN}{panel.panel_type.value:<12}{Colors.ENDC} (expected: {expected})")
        print()


async def demo_stats(jenny: JennyRuntime) -> None:
    """Demo 6: Runtime statistics."""
    print_header("Demo 6: Runtime Statistics")

    stats = jenny.get_stats()

    print(f"\n{Colors.BOLD}Jenny Runtime Statistics{Colors.ENDC}\n")

    print(f"  Panels created:    {Colors.GREEN}{stats['panels_created']}{Colors.ENDC}")
    print(f"  Actions executed:  {Colors.CYAN}{stats['actions_executed']}{Colors.ENDC}")
    print(f"  Queries processed: {Colors.BLUE}{stats['queries_processed']}{Colors.ENDC}")

    if stats.get("uptime_seconds"):
        uptime = stats["uptime_seconds"]
        print(f"  Uptime:            {Colors.DIM}{uptime:.2f}s{Colors.ENDC}")

    if stats.get("streaming"):
        stream = stats["streaming"]
        print(f"\n  {Colors.BOLD}Streaming:{Colors.ENDC}")
        print(f"    Total subscriptions: {stream.get('total_subscriptions', 0)}")
        print(f"    Active:              {stream.get('active_subscriptions', 0)}")
        print(f"    Updates sent:        {stream.get('total_updates_sent', 0)}")


# ============================================================================
# Interactive Mode
# ============================================================================

async def interactive_mode(jenny: JennyRuntime) -> None:
    """Interactive Jenny session."""
    print_header("Interactive Jenny Session")

    print(f"""
{Colors.BOLD}Commands:{Colors.ENDC}
  ask <question>   - Ask Jenny a question
  pin <id>         - Pin a panel
  dismiss <id>     - Dismiss a panel
  why <id>         - Ask 'Why this UI?'
  list             - List all panels
  history          - Show history
  stats            - Show statistics
  quit             - Exit

{Colors.DIM}Panel IDs can be shortened (first 8 chars){Colors.ENDC}
""")

    while True:
        try:
            user_input = input(f"\n{Colors.CYAN}jenny>{Colors.ENDC} ").strip()

            if not user_input:
                continue

            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command == "quit" or command == "exit":
                print_info("Goodbye!")
                break

            elif command == "ask":
                if not args:
                    print_warning("Usage: ask <question>")
                    continue

                panel = await jenny.ask(args)
                print_success(f"Created panel: {panel.id[:8]}")
                print_panel_card(panel)

            elif command == "pin":
                if not args:
                    print_warning("Usage: pin <panel_id>")
                    continue

                result = await jenny.act(args, "pin")
                if result.status == ActionStatus.SUCCESS:
                    print_success("Panel pinned!")
                else:
                    print_warning(f"Failed: {result.error}")

            elif command == "dismiss":
                if not args:
                    print_warning("Usage: dismiss <panel_id>")
                    continue

                result = await jenny.act(args, "dismiss")
                if result.status == ActionStatus.SUCCESS:
                    print_success("Panel dismissed!")
                else:
                    print_warning(f"Failed: {result.error}")

            elif command == "why":
                if not args:
                    print_warning("Usage: why <panel_id>")
                    continue

                result = await jenny.act(args, "why")
                if result.status == ActionStatus.SUCCESS:
                    why_data = result.data.get("why_panel_data", {})
                    print(f"\n{Colors.BOLD}Why:{Colors.ENDC}")
                    for reason in why_data.get("reasoning", []):
                        print(f"  • {reason}")
                else:
                    print_warning(f"Failed: {result.error}")

            elif command == "list":
                panels = await jenny.list_panels()
                if not panels:
                    print_info("No active panels")
                else:
                    for p in panels:
                        print_panel_card(p)

            elif command == "history":
                history = jenny.history(limit=10)
                for entry in history:
                    print(f"  {entry.get('event_type', 'unknown')}: {entry.get('spec_id', '')[:8]}")

            elif command == "stats":
                await demo_stats(jenny)

            else:
                print_warning(f"Unknown command: {command}")

        except KeyboardInterrupt:
            print_info("\nGoodbye!")
            break
        except Exception as e:
            print_warning(f"Error: {e}")


# ============================================================================
# Main Entry Point
# ============================================================================

async def run_all_demos() -> None:
    """Run all demos in sequence."""
    print(f"""
{Colors.BOLD}{Colors.HEADER}
     ╦╔═╗╔╗╔╔╗╔╦ ╦
     ║║╣ ║║║║║║╚╦╝
    ╚╝╚═╝╝╚╝╝╚╝ ╩
{Colors.ENDC}
{Colors.CYAN}    Generative UI Runtime{Colors.ENDC}
{Colors.DIM}    "Disposable pixels, durable decisions."{Colors.ENDC}
""")

    config = JennyConfig(
        enable_ledger=True,
        auto_cleanup=True,
        cleanup_interval_seconds=300.0,  # 5 minutes for demo
    )

    async with JennyRuntime(config) as jenny:
        await demo_basic_flow(jenny)
        await asyncio.sleep(0.5)

        await demo_lifecycle_states(jenny)
        await asyncio.sleep(0.5)

        await demo_provenance(jenny)
        await asyncio.sleep(0.5)

        await demo_why_panel(jenny)
        await asyncio.sleep(0.5)

        await demo_panel_types(jenny)
        await asyncio.sleep(0.5)

        await demo_stats(jenny)

        print_header("All Demos Complete!")
        print(f"""
{Colors.GREEN}Jenny Generative UI Runtime{Colors.ENDC}

{Colors.BOLD}Key Features Demonstrated:{Colors.ENDC}
  ✓ Zero-config panel generation
  ✓ Intelligent panel type selection
  ✓ Lifecycle state machine (NASCENT → STABLE → DISSOLVING → ARCHIVED)
  ✓ Complete provenance tracking
  ✓ 'Why this UI?' meta-panels
  ✓ Multi-target rendering (HTML, Terminal, JSON)

{Colors.BOLD}Why Jenny is Different:{Colors.ENDC}
  • UI is ephemeral, decisions are permanent
  • Every click logged with complete provenance
  • Self-explaining UI ("Why did you show me this?")
  • Streaming data bindings for live updates

{Colors.DIM}Run with --interactive for an interactive session{Colors.ENDC}
""")


async def main() -> None:
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        async with JennyRuntime() as jenny:
            await interactive_mode(jenny)
    else:
        await run_all_demos()


if __name__ == "__main__":
    asyncio.run(main())
