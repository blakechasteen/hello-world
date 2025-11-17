#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Query CLI
==================

Command-line interface for querying org-mode knowledge graphs.

Usage:
    python -m HoloLoom.query "What should I work on?"
    python -m HoloLoom.query "What's due this week?" --format json
    python -m HoloLoom.query "When did I finish auth refactor?" --kg my_knowledge.jsonl
"""

import asyncio
import argparse
import sys
from pathlib import Path

from HoloLoom.memory.graph import KG
from HoloLoom.query.engine import QueryEngine
from HoloLoom.query.formatter import format_result


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Query your org-mode knowledge graph with natural language',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m HoloLoom.query "What should I work on?"
  python -m HoloLoom.query "What's due this week?"
  python -m HoloLoom.query "When did I finish the auth refactor?"
  python -m HoloLoom.query "Show me my notes about neural networks"
  python -m HoloLoom.query "What am I currently working on?"
  python -m HoloLoom.query "Statistics" --format json
  python -m HoloLoom.query "What's due?" --format org
        """
    )

    parser.add_argument(
        'query',
        help='Natural language query'
    )

    parser.add_argument(
        '--kg',
        '--knowledge-graph',
        dest='kg_path',
        help='Path to knowledge graph file (JSONL format)',
        default=None
    )

    parser.add_argument(
        '--monitor',
        help='Path to org files directory for live monitoring',
        default=None
    )

    parser.add_argument(
        '--format',
        choices=['text', 'json', 'org'],
        default='text',
        help='Output format (default: text)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Load knowledge graph
    if args.kg_path:
        kg_path = Path(args.kg_path).expanduser()
        if not kg_path.exists():
            print(f"Error: Knowledge graph file not found: {kg_path}", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(f"Loading knowledge graph from: {kg_path}")

        kg = KG.load(str(kg_path))

        if args.verbose:
            stats = kg.stats()
            print(f"Loaded graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    else:
        # Create empty graph
        kg = KG()

        if args.verbose:
            print("No knowledge graph specified, using empty graph")
            print("Tip: Use --kg to specify a knowledge graph file")

    # Set up monitor if specified
    monitor = None
    if args.monitor:
        from HoloLoom.spinningWheel.orgmode_live import OrgLiveMonitor
        from HoloLoom.spinningWheel.base import SpinnerConfig

        monitor_path = Path(args.monitor).expanduser()
        if not monitor_path.exists():
            print(f"Error: Monitor directory not found: {monitor_path}", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(f"Setting up monitoring for: {monitor_path}")

        # Note: This is synchronous version for CLI
        # In async context, would call monitor.start()
        config = SpinnerConfig(enable_enrichment=False)
        monitor = OrgLiveMonitor(kg, watch_dir=str(monitor_path), config=config)

        # Load files synchronously for now
        if args.verbose:
            print("Loading org files...")

        asyncio.run(monitor._initial_load())

        if args.verbose:
            stats = kg.stats()
            print(f"After loading org files: {stats['num_nodes']} nodes, {stats['num_edges']} edges")

    # Create query engine
    engine = QueryEngine(kg, monitor)

    # Execute query
    if args.verbose:
        print(f"\nExecuting query: {args.query}")
        print("")

    try:
        result = asyncio.run(engine.query(args.query))

        # Format and print result
        formatted = format_result(result, format=args.format)
        print(formatted)

    except Exception as e:
        print(f"Error executing query: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
