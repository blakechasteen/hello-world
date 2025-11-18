#!/usr/bin/env python3
"""
Filesystem Ingestion CLI
=========================

Command-line interface for ingesting files from the filesystem into HoloLoom.

**Usage**:
    # Ingest a directory
    python -m HoloLoom.ingestion.filesystem /path/to/docs

    # With custom patterns
    python -m HoloLoom.ingestion.filesystem /path/to/docs \\
        --allow "*.md" "*.txt" \\
        --deny "*.log" ".git/**"

    # Incremental (only new/modified files)
    python -m HoloLoom.ingestion.filesystem /path/to/docs --incremental

    # Preview without ingesting
    python -m HoloLoom.ingestion.filesystem /path/to/docs --dry-run

**Features**:
    - Directory scanning with glob patterns
    - Incremental ingestion (mtime tracking)
    - Dry-run mode (preview without ingesting)
    - Progress reporting
    - Integration with HoloLoom memory system

Author: Claude Code
Date: November 2025
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional
import json

# Import HoloLoom components
try:
    from HoloLoom.spinningWheel.filesystem_spinner import FilesystemSpinner, FilesystemSpinnerConfig
    from HoloLoom.hololoom import HoloLoom
    from HoloLoom.config import Config
except ImportError as e:
    print(f"Error: Failed to import HoloLoom components: {e}", file=sys.stderr)
    print("Make sure you're running from the repository root with PYTHONPATH=.", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# CLI Functions
# ============================================================================

async def ingest_directory(
    directory: str,
    chunk_size: int,
    chunk_overlap: int,
    allow_patterns: List[str],
    deny_patterns: List[str],
    recursive: bool,
    incremental: bool,
    interactive: bool,
    dry_run: bool,
    checkpoint_dir: Optional[str],
    importance_threshold: float,
    output_format: str,
    integrate: bool,
):
    """
    Ingest directory and optionally add to HoloLoom memory.

    Args:
        directory: Directory to ingest
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks
        allow_patterns: Glob patterns to include
        deny_patterns: Glob patterns to exclude
        recursive: Scan subdirectories
        incremental: Only process new/modified files
        interactive: Interactively select files to ingest
        dry_run: Preview without ingesting
        checkpoint_dir: Directory for checkpoints
        importance_threshold: Minimum importance to include
        output_format: Output format (text, json)
        integrate: Integrate with HoloLoom memory
    """
    # Validate directory
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)
    if not dir_path.is_dir():
        print(f"Error: Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    # Create spinner
    spinner_config = FilesystemSpinnerConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        allow_patterns=allow_patterns,
        deny_patterns=deny_patterns
    )

    spinner = FilesystemSpinner(
        config=spinner_config,
        importance_threshold=importance_threshold,
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None
    )

    # Interactive mode
    if interactive:
        print(f"📁 Interactive File Selection")
        print(f"   Directory: {directory}")
        print(f"   Patterns: {', '.join(allow_patterns)}")
        print(f"   Exclude: {', '.join(deny_patterns)}")
        print()

        # Interactive selection
        selected_files = spinner.interactive_select_files(
            dir_path,
            recursive=recursive,
            follow_symlinks=False
        )

        if not selected_files:
            print("\n❌ No files selected or operation cancelled")
            sys.exit(0)

        # Process selected files
        try:
            result = await spinner.spin_custom_files(selected_files)
        except Exception as e:
            print(f"Error: Ingestion failed: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        # Non-interactive mode
        print(f"📁 Scanning: {directory}")
        print(f"   Patterns: {', '.join(allow_patterns)}")
        print(f"   Exclude: {', '.join(deny_patterns)}")
        print(f"   Mode: {'incremental' if incremental else 'full'}")
        print(f"   Chunk size: {chunk_size} chars (overlap: {chunk_overlap})")
        print()

        # Ingest
        try:
            if incremental:
                result = await spinner.spin_incremental(directory, recursive=recursive)
            else:
                result = await spinner.spin(directory, recursive=recursive)
        except Exception as e:
            print(f"Error: Ingestion failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Report results
    if not result.success:
        print(f"❌ Ingestion failed: {result.error_message}", file=sys.stderr)
        sys.exit(1)

    # Display summary
    if output_format == "json":
        # JSON output
        print(json.dumps(result.to_dict(), indent=2))
    else:
        # Human-readable output
        print(f"✅ Ingestion complete!")
        print(f"   Shards created: {result.shard_count}")
        print(f"   Entities extracted: {result.entity_count}")
        print(f"   Motifs extracted: {result.motif_count}")
        print(f"   Average importance: {result.avg_importance:.2f}")
        print(f"   Processing time: {result.processing_time_ms:.0f}ms")
        print(f"   Input size: {result.input_size_bytes:,} bytes")

        if result.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   - {warning}")

    # Dry run: stop here
    if dry_run:
        print(f"\n🔍 Dry run complete (shards not added to memory)")
        return

    # Integrate with HoloLoom memory
    if integrate:
        print(f"\n🧠 Adding to HoloLoom memory...")

        try:
            # Create HoloLoom instance
            config = Config.fast()
            async with HoloLoom(cfg=config) as loom:
                # Add each shard
                for shard in result.shards:
                    await loom.experience(shard.text)

                print(f"✅ Added {len(result.shards)} shards to memory")

                # Get metrics
                metrics = loom.get_metrics()
                print(f"   Active memories: {metrics['activation']['active_nodes']}")
                print(f"   Total memories: {metrics['activation']['total_nodes']}")

        except Exception as e:
            print(f"❌ Failed to integrate with HoloLoom: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"\nℹ️  Use --integrate to add shards to HoloLoom memory")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest files from filesystem into HoloLoom",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest Markdown and text files
  python -m HoloLoom.ingestion.filesystem /path/to/docs

  # Custom patterns
  python -m HoloLoom.ingestion.filesystem /path/to/docs \\
      --allow "*.md" "*.rst" --deny "build/**"

  # Interactive file selection
  python -m HoloLoom.ingestion.filesystem /path/to/docs --interactive

  # Incremental (only new/modified files)
  python -m HoloLoom.ingestion.filesystem /path/to/docs --incremental

  # Dry run (preview without ingesting)
  python -m HoloLoom.ingestion.filesystem /path/to/docs --dry-run

  # Integrate with HoloLoom memory
  python -m HoloLoom.ingestion.filesystem /path/to/docs --integrate

  # JSON output for scripting
  python -m HoloLoom.ingestion.filesystem /path/to/docs --output json

For more information: https://github.com/blakechasteen/hello-world
        """
    )

    # Required arguments
    parser.add_argument(
        "directory",
        help="Directory to ingest"
    )

    # File filtering
    parser.add_argument(
        "--allow",
        nargs="+",
        default=["*.txt", "*.md"],
        help="Glob patterns to include (default: *.txt *.md)"
    )
    parser.add_argument(
        "--deny",
        nargs="+",
        default=["*.log", ".git/**", "node_modules/**", ".venv/**", "__pycache__/**"],
        help="Glob patterns to exclude"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't scan subdirectories"
    )

    # Chunking
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Characters per chunk (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)"
    )

    # Incremental ingestion
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only process new/modified files (uses checkpoints)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Directory for checkpoints (default: ~/.hololoom/checkpoints)"
    )

    # Interactive selection
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively select files to ingest"
    )

    # Importance filtering
    parser.add_argument(
        "--importance-threshold",
        type=float,
        default=0.0,
        help="Minimum importance score to include (0.0-1.0, default: 0.0)"
    )

    # Integration
    parser.add_argument(
        "--integrate",
        action="store_true",
        help="Add shards to HoloLoom memory after ingestion"
    )

    # Output
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without ingesting (shows what would be created)"
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Run ingestion
    asyncio.run(
        ingest_directory(
            directory=args.directory,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            allow_patterns=args.allow,
            deny_patterns=args.deny,
            recursive=not args.no_recursive,
            incremental=args.incremental,
            interactive=args.interactive,
            dry_run=args.dry_run,
            checkpoint_dir=args.checkpoint_dir,
            importance_threshold=args.importance_threshold,
            output_format=args.output,
            integrate=args.integrate,
        )
    )


if __name__ == "__main__":
    main()
