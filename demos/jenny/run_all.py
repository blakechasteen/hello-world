#!/usr/bin/env python3
"""
Jenny Moonshot: Unified Demo Runner
===================================

Runs all Jenny Moonshot demos with dependency resolution
and comprehensive result reporting.

Usage:
    # Run all demos
    PYTHONPATH=. python demos/jenny/run_all.py

    # Run specific phase
    PYTHONPATH=. python demos/jenny/run_all.py --phase m1

    # Run by tag
    PYTHONPATH=. python demos/jenny/run_all.py --tag core

    # List available phases
    PYTHONPATH=. python demos/jenny/run_all.py --list

Author: HoloLoom Team
Date: 2025-12-09 (Jenny Moonshot Extensibility)
"""

import argparse
import asyncio
import sys
import time
from typing import List, Optional

# Import the demo infrastructure
from demos.jenny import (
    discover_and_register,
    run_all_demos,
    run_demo,
    run_demos_by_tag,
    get_registry,
    get_registry_summary,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_demo_summary,
)


def print_banner():
    """Print Jenny Moonshot banner."""
    banner = """
    +==============================================================+
    |                                                              |
    |   Jenny Moonshot: Extensible Demo System                     |
    |   ------------------------------------------                 |
    |                                                              |
    |   Phase 1: Extensibility Foundation                          |
    |   * Plugin-based demo discovery                              |
    |   * Dependency resolution                                    |
    |   * Tag-based filtering                                      |
    |                                                              |
    +==============================================================+
    """
    print(banner)


def list_phases():
    """List all registered phases."""
    registry = get_registry()
    phases = registry.list_phases()

    print_header("Registered Demo Phases")
    print()

    if not phases:
        print_warning("No phases registered. Run discover_and_register() first.")
        return

    # Group by tags
    by_tag = {}
    for phase in phases:
        for tag in phase.tags or ['untagged']:
            if tag not in by_tag:
                by_tag[tag] = []
            by_tag[tag].append(phase)

    for tag in sorted(by_tag.keys()):
        print(f"  [{tag}]")
        for phase in sorted(by_tag[tag], key=lambda p: p.order):
            deps = f" (requires: {', '.join(phase.requires)})" if phase.requires else ""
            print(f"    • {phase.id}: {phase.name}{deps}")
            if phase.description:
                print(f"        {phase.description}")
        print()

    print(f"  Total: {len(phases)} phases")


def list_tags():
    """List all available tags."""
    registry = get_registry()
    tags = registry.list_tags()

    print_header("Available Tags")
    print()

    for tag in sorted(tags):
        count = len([p for p in registry.list_phases() if tag in (p.tags or [])])
        print(f"  • {tag}: {count} phase(s)")

    print()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Jenny Moonshot Demo Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py              # Run all demos
  python run_all.py --phase m1   # Run specific phase
  python run_all.py --tag core   # Run phases with tag
  python run_all.py --list       # List available phases
  python run_all.py --tags       # List available tags
        """
    )

    parser.add_argument(
        "--phase", "-p",
        help="Run specific phase by ID (e.g., m1, m2)"
    )
    parser.add_argument(
        "--tag", "-t",
        help="Run phases with specific tag (e.g., core, output)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all registered phases"
    )
    parser.add_argument(
        "--tags",
        action="store_true",
        help="List all available tags"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip the banner"
    )

    args = parser.parse_args()

    # Print banner unless suppressed
    if not args.no_banner:
        print_banner()

    # Discover and register demo modules
    print_info("Discovering demo modules...")
    discovered = discover_and_register()
    print_success(f"Registered {discovered} demo phases")
    print()

    # Handle list commands
    if args.list:
        list_phases()
        return 0

    if args.tags:
        list_tags()
        return 0

    # Run demos
    start_time = time.perf_counter()

    if args.phase:
        # Run specific phase
        print_header(f"Running Phase: {args.phase}")
        result = await run_demo(args.phase)
        results_list = [result]
        results_summary = {"results": results_list}

    elif args.tag:
        # Run by tag
        print_header(f"Running Phases with Tag: {args.tag}")
        results_summary = await run_demos_by_tag(args.tag)
        results_list = results_summary.get("results", [])

    else:
        # Run all demos
        print_header("Running All Demo Phases")
        results_summary = await run_all_demos()
        results_list = results_summary.get("results", [])

    total_duration = time.perf_counter() - start_time

    # Print summary
    print()
    print_demo_summary(results_list)

    # Overall timing
    print(f"\n  Total Execution Time: {total_duration * 1000:.1f}ms")

    # Return exit code based on results
    failed = sum(1 for r in results_list
                 if r.get('status') not in ('success', 'skipped'))
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
