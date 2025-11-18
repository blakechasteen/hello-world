#!/usr/bin/env python3
"""
Interactive File Selection Demo
================================

Demonstrates interactive file selection feature for FilesystemSpinner.

**Usage**:
    PYTHONPATH=. python demos/demo_interactive_selection.py

Author: Claude Code
Date: November 2025
"""

import asyncio
import tempfile
from pathlib import Path

# Import directly to avoid dependency issues
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "filesystem_spinner",
    Path(__file__).parent.parent / "HoloLoom" / "spinningWheel" / "filesystem_spinner.py"
)
filesystem_module = importlib.util.module_from_spec(spec)

# Import types first
from HoloLoom.Documentation.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner, SpinnerCapabilities, SpinnerStatus, SpinResult,
    SpinnerCheckpoint, ImportanceSignals, ImportanceScore
)

# Now load the module
spec.loader.exec_module(filesystem_module)
FilesystemSpinner = filesystem_module.FilesystemSpinner


async def main():
    """Run interactive selection demo."""
    print("=" * 70)
    print(" " * 15 + "Interactive File Selection Demo")
    print("=" * 70)
    print()
    print("This demo shows how to interactively select files for ingestion.")
    print()

    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Create test files with varying importance
        print("Creating test files...")

        # High importance: README
        (base / "README.md").write_text("""# Important Project

This is the main README file for the project.

## Features

- Feature 1
- Feature 2
- Feature 3

## Getting Started

Follow these instructions...
""")

        # Medium importance: Guide
        (base / "guide.md").write_text("""# User Guide

This guide explains how to use the system.

## Step 1

Do this first.

## Step 2

Then do this.
""")

        # Lower importance: Notes
        (base / "notes.txt").write_text("""Random notes:

- Remember to check this
- Don't forget that
- TODO: Fix the thing
""")

        # Very low importance: Small file
        (base / "temp.txt").write_text("Just a temp file.")

        # Another medium file
        (base / "tutorial.md").write_text("""# Tutorial

Learn how to get started with this project.

Follow along with these examples...
""")

        print(f"Created 5 test files in {base}")
        print()

        # Create spinner
        spinner = FilesystemSpinner(
            allow_patterns=["*.md", "*.txt"],
            deny_patterns=[]
        )

        print("Starting interactive file selection...")
        print("(This would normally be interactive, but running in automated mode for demo)")
        print()

        # In a real scenario, this would be interactive
        # For the demo, we'll just scan and show what would be displayed
        all_files = spinner._scan_directory(base, recursive=False, follow_symlinks=False)

        print("Files that would be shown in interactive mode:")
        print()
        print(f"{'#':<4} {'File':<30} {'Size':<10} {'Importance':<12}")
        print("-" * 56)

        for i, file_path in enumerate(sorted(all_files), 1):
            importance = spinner.score_importance(file_path)
            size = file_path.stat().st_size
            size_str = spinner._format_size(size)

            print(f"{i:<4} {file_path.name:<30} {size_str:<10} {importance.score:.2f}")

        print()
        print("In interactive mode, you could:")
        print("  - Type a number to toggle file selection")
        print("  - Type 'all' to select all files")
        print("  - Type 'none' to deselect all")
        print("  - Type 'invert' to invert selection")
        print("  - Type 'done' to proceed with selected files")
        print("  - Type 'cancel' to exit")
        print()

        print("To try interactive mode yourself, run:")
        print()
        print("  python -m HoloLoom.ingestion.filesystem /path/to/docs --interactive")
        print()


if __name__ == "__main__":
    asyncio.run(main())
