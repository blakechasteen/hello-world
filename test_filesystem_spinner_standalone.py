#!/usr/bin/env python3
"""
Standalone test for FilesystemSpinner (no dependencies).

Tests the core functionality without requiring HoloLoom imports.
"""

import asyncio
import tempfile
from pathlib import Path
import sys

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Import directly to avoid numpy dependency in __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "filesystem_spinner",
    Path(__file__).parent / "HoloLoom" / "spinningWheel" / "filesystem_spinner.py"
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


async def test_basic():
    """Test basic ingestion."""
    print("Test 1: Basic ingestion")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Create test files
        (base / "README.md").write_text("""# Test Project

This is a test README file.
It has multiple lines of content.

## Section 1

Content here.

## Section 2

More content.
""")

        (base / "guide.txt").write_text("""Quick Start

1. First step
2. Second step
3. Third step
""")

        # Create log file (should be excluded)
        (base / "debug.log").write_text("Log line 1\nLog line 2\n")

        print(f"Created test directory: {base}")
        print(f"Files: README.md, guide.txt, debug.log")
        print()

        # Ingest
        spinner = FilesystemSpinner(
            allow_patterns=["*.md", "*.txt"],
            deny_patterns=["*.log"]
        )

        result = await spinner.spin(str(base))

        # Check results
        assert result.success, "Ingestion failed"
        assert result.shard_count > 0, "No shards created"

        print(f"✅ Success!")
        print(f"   Shards: {result.shard_count}")
        print(f"   Processing time: {result.processing_time_ms:.0f}ms")
        print()

        # Show shards
        file_names = {s.metadata["file_name"] for s in result.shards}
        print(f"   Files ingested: {', '.join(file_names)}")
        assert "README.md" in file_names
        assert "guide.txt" in file_names
        assert "debug.log" not in file_names

        print(f"   ✓ Correctly excluded debug.log")
        print()


async def test_chunking():
    """Test chunking."""
    print("Test 2: Chunking")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Create large file
        large_content = "This is a sentence. " * 100  # ~2000 chars
        (base / "large.txt").write_text(large_content)

        print(f"Created large file (~{len(large_content)} chars)")
        print()

        # Ingest with small chunk size
        spinner = FilesystemSpinner(
            chunk_size=500,
            chunk_overlap=100,
            allow_patterns=["*.txt"]
        )

        result = await spinner.spin(str(base))

        assert result.success
        assert result.shard_count >= 3, "Should create multiple chunks"

        print(f"✅ Success!")
        print(f"   Chunks created: {result.shard_count}")
        print(f"   Chunk size: 500 chars (overlap: 100)")
        print()


async def test_incremental():
    """Test incremental ingestion."""
    print("Test 3: Incremental ingestion")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        checkpoint_dir = base / "checkpoints"
        checkpoint_dir.mkdir()

        # Create test file
        (base / "test.txt").write_text("Original content")

        print("First run (all files)...")
        spinner = FilesystemSpinner(
            allow_patterns=["*.txt"],
            checkpoint_dir=checkpoint_dir
        )

        result1 = await spinner.spin_incremental(str(base))
        print(f"   Shards: {result1.shard_count}")
        assert result1.shard_count > 0

        # Second run (no changes)
        print("\nSecond run (no changes)...")
        result2 = await spinner.spin_incremental(str(base))
        print(f"   Shards: {result2.shard_count}")
        assert result2.shard_count == 0, "Should skip unchanged files"

        # Modify file
        print("\nModifying test.txt...")
        import time
        time.sleep(0.1)
        (base / "test.txt").write_text("Modified content")

        # Third run (only modified)
        print("Third run (modified file)...")
        result3 = await spinner.spin_incremental(str(base))
        print(f"   Shards: {result3.shard_count}")
        assert result3.shard_count > 0, "Should process modified file"

        print(f"\n✅ Success!")
        print(f"   ✓ Incremental ingestion working correctly")
        print()


async def test_metadata():
    """Test shard metadata."""
    print("Test 4: Metadata")
    print("-" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        (base / "test.md").write_text("# Test\n\nContent here.")

        spinner = FilesystemSpinner(allow_patterns=["*.md"])
        result = await spinner.spin(str(base))

        assert result.success
        shard = result.shards[0]

        print("Checking metadata fields...")
        required_fields = [
            "file_path", "file_name", "file_size", "mtime",
            "media_type", "chunk_index", "total_chunks",
            "importance_score", "importance_reason"
        ]

        for field in required_fields:
            assert field in shard.metadata, f"Missing field: {field}"
            print(f"   ✓ {field}: {shard.metadata[field]}")

        print(f"\n✅ Success!")
        print(f"   All metadata fields present")
        print()


async def main():
    """Run all tests."""
    print("=" * 70)
    print(" " * 15 + "FilesystemSpinner Standalone Tests")
    print("=" * 70)
    print()

    try:
        await test_basic()
        await test_chunking()
        await test_incremental()
        await test_metadata()

        print("=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
        print()

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
