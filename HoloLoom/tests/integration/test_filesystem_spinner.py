"""
Integration Tests for FilesystemSpinner
========================================

Tests filesystem ingestion functionality.

**Run**:
    pytest HoloLoom/tests/integration/test_filesystem_spinner.py -v

Author: Claude Code
Date: November 2025
"""

import pytest
import tempfile
from pathlib import Path
import time

from HoloLoom.spinningWheel.filesystem_spinner import (
    FilesystemSpinner,
    FilesystemSpinnerConfig,
    ingest_directory,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_docs_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Create test files
        (base / "README.md").write_text("""# Test Project

This is a test project for filesystem ingestion.

## Features

- Feature 1
- Feature 2
""")

        (base / "guide.txt").write_text("""Quick Start Guide

1. Install dependencies
2. Run the application
3. Check the results
""")

        # Create subdirectory
        subdir = base / "docs"
        subdir.mkdir()
        (subdir / "advanced.md").write_text("""# Advanced Topics

This covers advanced usage.
""")

        # Create excluded file
        (base / "debug.log").write_text("Log entry 1\nLog entry 2\n")

        # Create .git directory (should be excluded)
        gitdir = base / ".git"
        gitdir.mkdir()
        (gitdir / "config").write_text("[core]\n")

        yield base


@pytest.fixture
def checkpoint_dir():
    """Create a temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Tests: Discovery
# ============================================================================

@pytest.mark.asyncio
async def test_discover_files_basic(temp_docs_dir):
    """Test basic file discovery."""
    spinner = FilesystemSpinner(
        allow_patterns=["*.md", "*.txt"],
        deny_patterns=["*.log", ".git/**"]
    )

    result = await spinner.spin(str(temp_docs_dir))

    assert result.success
    assert result.shard_count > 0

    # Check that .md and .txt files were included
    file_names = {shard.metadata["file_name"] for shard in result.shards}
    assert "README.md" in file_names
    assert "guide.txt" in file_names

    # Check that .log was excluded
    assert "debug.log" not in file_names


@pytest.mark.asyncio
async def test_discover_files_recursive(temp_docs_dir):
    """Test recursive directory scanning."""
    spinner = FilesystemSpinner(
        allow_patterns=["*.md"],
    )

    result = await spinner.spin(str(temp_docs_dir), recursive=True)

    assert result.success

    # Should find files in subdirectory
    file_names = {shard.metadata["file_name"] for shard in result.shards}
    assert "advanced.md" in file_names


@pytest.mark.asyncio
async def test_discover_files_non_recursive(temp_docs_dir):
    """Test non-recursive scanning (only top-level)."""
    spinner = FilesystemSpinner(
        allow_patterns=["*.md"],
    )

    result = await spinner.spin(str(temp_docs_dir), recursive=False)

    assert result.success

    # Should NOT find files in subdirectory
    file_names = {shard.metadata["file_name"] for shard in result.shards}
    assert "README.md" in file_names
    assert "advanced.md" not in file_names


# ============================================================================
# Tests: Chunking
# ============================================================================

@pytest.mark.asyncio
async def test_chunking_small_file(temp_docs_dir):
    """Test that small files create single chunk."""
    # Create small file
    small_file = temp_docs_dir / "small.txt"
    small_file.write_text("This is a small file.")

    spinner = FilesystemSpinner(
        chunk_size=1000,
        allow_patterns=["small.txt"]
    )

    result = await spinner.spin(str(temp_docs_dir))

    assert result.success

    # Small file should create 1 chunk
    small_shards = [s for s in result.shards if s.metadata["file_name"] == "small.txt"]
    assert len(small_shards) == 1
    assert small_shards[0].metadata["chunk_index"] == 0
    assert small_shards[0].metadata["total_chunks"] == 1


@pytest.mark.asyncio
async def test_chunking_large_file(temp_docs_dir):
    """Test that large files are chunked."""
    # Create large file (over 1000 chars)
    large_content = "This is a sentence. " * 100  # ~2000 chars
    large_file = temp_docs_dir / "large.txt"
    large_file.write_text(large_content)

    spinner = FilesystemSpinner(
        chunk_size=1000,
        chunk_overlap=100,
        allow_patterns=["large.txt"]
    )

    result = await spinner.spin(str(temp_docs_dir))

    assert result.success

    # Large file should create multiple chunks
    large_shards = [s for s in result.shards if s.metadata["file_name"] == "large.txt"]
    assert len(large_shards) >= 2
    assert large_shards[0].metadata["total_chunks"] >= 2


@pytest.mark.asyncio
async def test_chunk_overlap(temp_docs_dir):
    """Test that chunks have proper overlap."""
    content = "A" * 500 + "MARKER" + "B" * 500
    test_file = temp_docs_dir / "overlap.txt"
    test_file.write_text(content)

    spinner = FilesystemSpinner(
        chunk_size=600,
        chunk_overlap=100,
        allow_patterns=["overlap.txt"]
    )

    result = await spinner.spin(str(temp_docs_dir))

    assert result.success

    shards = [s for s in result.shards if s.metadata["file_name"] == "overlap.txt"]
    assert len(shards) >= 2

    # MARKER should appear in first chunk
    assert "MARKER" in shards[0].text


# ============================================================================
# Tests: Incremental Ingestion
# ============================================================================

@pytest.mark.asyncio
async def test_incremental_first_run(temp_docs_dir, checkpoint_dir):
    """Test first incremental run (processes all files)."""
    spinner = FilesystemSpinner(
        allow_patterns=["*.md", "*.txt"],
        checkpoint_dir=checkpoint_dir
    )

    result = await spinner.spin_incremental(str(temp_docs_dir))

    assert result.success
    assert result.shard_count > 0


@pytest.mark.asyncio
async def test_incremental_no_changes(temp_docs_dir, checkpoint_dir):
    """Test incremental run with no file changes (processes nothing)."""
    spinner = FilesystemSpinner(
        allow_patterns=["*.md", "*.txt"],
        checkpoint_dir=checkpoint_dir
    )

    # First run
    result1 = await spinner.spin_incremental(str(temp_docs_dir))
    count1 = result1.shard_count

    # Second run (no changes)
    time.sleep(0.1)
    result2 = await spinner.spin_incremental(str(temp_docs_dir))

    assert result2.success
    assert result2.shard_count == 0
    assert "0 new/modified" in result2.warnings[0]


@pytest.mark.asyncio
async def test_incremental_modified_file(temp_docs_dir, checkpoint_dir):
    """Test incremental run with modified file (processes only modified)."""
    spinner = FilesystemSpinner(
        allow_patterns=["*.md", "*.txt"],
        checkpoint_dir=checkpoint_dir
    )

    # First run
    result1 = await spinner.spin_incremental(str(temp_docs_dir))

    # Modify one file
    time.sleep(0.1)
    readme = temp_docs_dir / "README.md"
    current = readme.read_text()
    readme.write_text(current + "\n\n## New Section\n")

    # Second run
    result2 = await spinner.spin_incremental(str(temp_docs_dir))

    assert result2.success
    assert result2.shard_count > 0

    # Only README.md should be processed
    file_names = {shard.metadata["file_name"] for shard in result2.shards}
    assert file_names == {"README.md"}


# ============================================================================
# Tests: Importance Scoring
# ============================================================================

@pytest.mark.asyncio
async def test_importance_scoring(temp_docs_dir):
    """Test that importance scoring works."""
    spinner = FilesystemSpinner(
        allow_patterns=["*.md", "*.txt"]
    )

    result = await spinner.spin(str(temp_docs_dir))

    assert result.success

    # All shards should have importance metadata
    for shard in result.shards:
        assert "importance_score" in shard.metadata
        assert 0.0 <= shard.metadata["importance_score"] <= 1.0
        assert "importance_reason" in shard.metadata


@pytest.mark.asyncio
async def test_importance_threshold(temp_docs_dir):
    """Test filtering by importance threshold."""
    # Create low-importance file
    temp_docs_dir.joinpath("temp.txt").write_text("x")

    # Low threshold (include all)
    spinner1 = FilesystemSpinner(
        allow_patterns=["*.txt"],
        importance_threshold=0.0
    )
    result1 = await spinner1.spin(str(temp_docs_dir))

    # High threshold (filter out low-importance)
    spinner2 = FilesystemSpinner(
        allow_patterns=["*.txt"],
        importance_threshold=0.8
    )
    result2 = await spinner2.spin(str(temp_docs_dir))

    # High threshold should filter out some shards
    assert result1.shard_count >= result2.shard_count


# ============================================================================
# Tests: Metadata
# ============================================================================

@pytest.mark.asyncio
async def test_shard_metadata(temp_docs_dir):
    """Test that shards have correct metadata."""
    spinner = FilesystemSpinner(
        allow_patterns=["*.md"]
    )

    result = await spinner.spin(str(temp_docs_dir))

    assert result.success

    # Check README.md shard
    readme_shards = [s for s in result.shards if s.metadata["file_name"] == "README.md"]
    assert len(readme_shards) > 0

    shard = readme_shards[0]

    # Check required metadata
    assert "file_path" in shard.metadata
    assert "file_name" in shard.metadata
    assert shard.metadata["file_name"] == "README.md"
    assert "file_size" in shard.metadata
    assert "mtime" in shard.metadata
    assert "media_type" in shard.metadata
    assert shard.metadata["media_type"] == "text/markdown"
    assert "chunk_index" in shard.metadata
    assert "total_chunks" in shard.metadata


# ============================================================================
# Tests: Convenience Function
# ============================================================================

@pytest.mark.asyncio
async def test_convenience_function(temp_docs_dir):
    """Test ingest_directory convenience function."""
    result = await ingest_directory(
        str(temp_docs_dir),
        allow_patterns=["*.md"],
        incremental=False
    )

    assert result.success
    assert result.shard_count > 0


# ============================================================================
# Tests: Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_nonexistent_directory():
    """Test that nonexistent directory raises error."""
    spinner = FilesystemSpinner()

    with pytest.raises(FileNotFoundError):
        await spinner.spin("/nonexistent/directory")


@pytest.mark.asyncio
async def test_file_instead_of_directory(temp_docs_dir):
    """Test that file path (not directory) raises error."""
    spinner = FilesystemSpinner()

    file_path = temp_docs_dir / "README.md"

    with pytest.raises(ValueError):
        await spinner.spin(str(file_path))
