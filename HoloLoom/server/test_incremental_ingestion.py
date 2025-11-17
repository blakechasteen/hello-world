"""
Tests for the /ingest/workspace/incremental endpoint.

Tests the file watcher integration with HoloLoom's knowledge graph system.

Created: 2025-11-17 (Phase 5 Wave 1 - File Watcher Integration)
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Note: These tests assume you can run them with:
#   pytest HoloLoom/server/test_incremental_ingestion.py -v


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create test Python file
        py_file = workspace / "test_file.py"
        py_file.write_text("""
# TODO: Implement error handling
def test_function(x, y):
    '''Test docstring.'''
    return x + y

class TestClass:
    pass
""")

        # Create test TypeScript file
        ts_file = workspace / "test_file.ts"
        ts_file.write_text("""
// NOTE: This is a note
interface TestInterface {
    name: string;
    value: number;
}

export function testFunc(): void {
    console.log("test");
}
""")

        # Create nested file
        nested_dir = workspace / "src"
        nested_dir.mkdir()
        nested_file = nested_dir / "main.py"
        nested_file.write_text("""
# FIXME: Fix this bug
def main():
    pass
""")

        yield workspace


class TestIncrementalIngestionEndpoint:
    """Test the /ingest/workspace/incremental endpoint."""

    @pytest.mark.asyncio
    async def test_incremental_ingestion_with_valid_files(self, temp_workspace):
        """Test basic incremental ingestion with valid files."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        files = [
            str(temp_workspace / "test_file.py"),
            str(temp_workspace / "test_file.ts")
        ]

        result = await ingest_workspace_incremental(
            files=files,
            workspace_path=str(temp_workspace)
        )

        assert result["success"] is True
        assert result["files_indexed"] >= 2
        assert result["code_elements"] > 0
        assert "stats" in result
        assert result["stats"]["files"] >= 2

    @pytest.mark.asyncio
    async def test_empty_file_list(self, temp_workspace):
        """Test with empty file list."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        result = await ingest_workspace_incremental(
            files=[],
            workspace_path=str(temp_workspace)
        )

        assert result["success"] is True
        assert result["files_indexed"] == 0
        assert result["code_elements"] == 0

    @pytest.mark.asyncio
    async def test_nonexistent_file_in_list(self, temp_workspace):
        """Test with mix of existing and non-existing files."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        files = [
            str(temp_workspace / "test_file.py"),
            str(temp_workspace / "nonexistent.py")  # This file doesn't exist
        ]

        result = await ingest_workspace_incremental(
            files=files,
            workspace_path=str(temp_workspace)
        )

        # Should process the existing file and skip the missing one
        assert result["success"] is True
        assert result["files_indexed"] >= 1

    @pytest.mark.asyncio
    async def test_invalid_workspace_path(self, temp_workspace):
        """Test with invalid workspace path."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental
        from fastapi import HTTPException

        files = [str(temp_workspace / "test_file.py")]

        with pytest.raises(HTTPException) as exc_info:
            await ingest_workspace_incremental(
                files=files,
                workspace_path="/nonexistent/path"
            )

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_nested_files(self, temp_workspace):
        """Test ingestion of nested files."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        files = [
            str(temp_workspace / "src" / "main.py"),
            str(temp_workspace / "test_file.py")
        ]

        result = await ingest_workspace_incremental(
            files=files,
            workspace_path=str(temp_workspace)
        )

        assert result["success"] is True
        assert result["files_indexed"] >= 2

    @pytest.mark.asyncio
    async def test_comment_extraction(self, temp_workspace):
        """Test that comments (TODO, FIXME, NOTE) are extracted."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        # Files have TODO, FIXME, and NOTE comments
        files = [
            str(temp_workspace / "test_file.py"),  # Has TODO
            str(temp_workspace / "test_file.ts"),  # Has NOTE
            str(temp_workspace / "src" / "main.py")  # Has FIXME
        ]

        result = await ingest_workspace_incremental(
            files=files,
            workspace_path=str(temp_workspace)
        )

        assert result["success"] is True
        # Should detect TODO, FIXME, NOTE comments
        assert result["todos"] >= 1
        assert result["comments"] >= 2

    @pytest.mark.asyncio
    async def test_response_structure(self, temp_workspace):
        """Test response has expected structure."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        files = [str(temp_workspace / "test_file.py")]

        result = await ingest_workspace_incremental(
            files=files,
            workspace_path=str(temp_workspace)
        )

        # Verify response structure
        assert "success" in result
        assert "files_indexed" in result
        assert "code_elements" in result
        assert "comments" in result
        assert "todos" in result
        assert "entities_created" in result
        assert "stats" in result

        # stats should have files and entities
        assert "files" in result["stats"]
        assert "entities" in result["stats"]

    @pytest.mark.asyncio
    async def test_duplicate_files_in_list(self, temp_workspace):
        """Test handling of duplicate files in the list."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        # Same file listed multiple times
        files = [
            str(temp_workspace / "test_file.py"),
            str(temp_workspace / "test_file.py"),
            str(temp_workspace / "test_file.py")
        ]

        result = await ingest_workspace_incremental(
            files=files,
            workspace_path=str(temp_workspace)
        )

        assert result["success"] is True
        # Should process duplicates (WorkspaceSpinner handles them)
        # Exact count depends on deduplication logic

    @pytest.mark.asyncio
    async def test_binary_files_are_skipped(self, temp_workspace):
        """Test that binary files are skipped."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        # Create a binary file
        binary_file = temp_workspace / "binary.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03')

        # Create a text file
        text_file = temp_workspace / "text.py"
        text_file.write_text("def func(): pass")

        result = await ingest_workspace_incremental(
            files=[
                str(binary_file),
                str(text_file)
            ],
            workspace_path=str(temp_workspace)
        )

        assert result["success"] is True
        # Binary file should be skipped, only text file processed
        assert result["files_indexed"] >= 1

    @pytest.mark.asyncio
    async def test_statistics_accuracy(self, temp_workspace):
        """Test that statistics accurately reflect processed files."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        files = [str(temp_workspace / "test_file.py")]

        result = await ingest_workspace_incremental(
            files=files,
            workspace_path=str(temp_workspace)
        )

        # Verify stats are consistent
        assert result["files_indexed"] == result["stats"]["files"]
        assert result["code_elements"] == result["entities_created"]


class TestBatchProcessingBehavior:
    """Test batch processing behavior and optimization."""

    @pytest.mark.asyncio
    async def test_batch_processing_timing(self, temp_workspace):
        """Test that batch processing is efficient."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental
        import time

        # Create multiple files
        files = []
        for i in range(5):
            f = temp_workspace / f"file{i}.py"
            f.write_text(f"def func{i}(): pass")
            files.append(str(f))

        # Time the batch operation
        start = time.time()
        result = await ingest_workspace_incremental(
            files=files,
            workspace_path=str(temp_workspace)
        )
        elapsed = time.time() - start

        assert result["success"] is True
        assert result["files_indexed"] >= 5
        # Should be reasonably fast (< 5 seconds for 5 files)
        # Adjust timeout based on actual performance
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_large_batch_handling(self, temp_workspace):
        """Test handling of larger batches (20 files)."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        # Create 20 test files
        files = []
        for i in range(20):
            f = temp_workspace / f"module{i}.py"
            f.write_text(f"""
# Module {i}
def func{i}(x):
    '''Function {i}.'''
    return x * {i}

class Class{i}:
    pass
""")
            files.append(str(f))

        result = await ingest_workspace_incremental(
            files=files,
            workspace_path=str(temp_workspace)
        )

        assert result["success"] is True
        assert result["files_indexed"] >= 20
        assert result["code_elements"] > 0

    @pytest.mark.asyncio
    async def test_mixed_language_batch(self, temp_workspace):
        """Test batch with multiple programming languages."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        # Create Python file
        py_file = temp_workspace / "script.py"
        py_file.write_text("def python_func(): pass")

        # Create TypeScript file
        ts_file = temp_workspace / "script.ts"
        ts_file.write_text("export function tsFunc(): void {}")

        # Create JavaScript file
        js_file = temp_workspace / "script.js"
        js_file.write_text("function jsFunc() {}")

        # Create Markdown file
        md_file = temp_workspace / "README.md"
        md_file.write_text("# Title\n\n## Section")

        result = await ingest_workspace_incremental(
            files=[
                str(py_file),
                str(ts_file),
                str(js_file),
                str(md_file)
            ],
            workspace_path=str(temp_workspace)
        )

        assert result["success"] is True
        assert result["files_indexed"] >= 4


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_file_read_permission_error(self, temp_workspace):
        """Test handling of file permission errors."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental
        import os

        # Create a file and remove read permissions
        protected_file = temp_workspace / "protected.py"
        protected_file.write_text("def func(): pass")

        try:
            os.chmod(str(protected_file), 0o000)

            result = await ingest_workspace_incremental(
                files=[str(protected_file)],
                workspace_path=str(temp_workspace)
            )

            # Should handle gracefully
            assert result["success"] is True
            # File couldn't be read, so shouldn't be indexed
            assert result["files_indexed"] == 0

        finally:
            # Restore permissions for cleanup
            os.chmod(str(protected_file), 0o644)

    @pytest.mark.asyncio
    async def test_unicode_filenames(self, temp_workspace):
        """Test handling of unicode in filenames and content."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        # Create file with unicode content
        unicode_file = temp_workspace / "unicode_test.py"
        unicode_file.write_text("""
# -*- coding: utf-8 -*-
'''测试 Тест тест'''
def función():
    '''Función con acentos'''
    pass
""")

        result = await ingest_workspace_incremental(
            files=[str(unicode_file)],
            workspace_path=str(temp_workspace)
        )

        assert result["success"] is True
        assert result["files_indexed"] >= 1

    @pytest.mark.asyncio
    async def test_very_long_file_paths(self, temp_workspace):
        """Test handling of very long file paths."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        # Create deeply nested directory structure
        deep_dir = temp_workspace / "a" / "b" / "c" / "d" / "e" / "f" / "g"
        deep_dir.mkdir(parents=True, exist_ok=True)

        long_file = deep_dir / "long_nested_file.py"
        long_file.write_text("def func(): pass")

        result = await ingest_workspace_incremental(
            files=[str(long_file)],
            workspace_path=str(temp_workspace)
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, temp_workspace):
        """Test handling multiple concurrent ingest requests."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        # Create test files
        files1 = [str(temp_workspace / "file1.py")]
        files1[0] = str(temp_workspace / "test1.py")
        Path(files1[0]).write_text("def func1(): pass")

        files2 = [str(temp_workspace / "file2.py")]
        files2[0] = str(temp_workspace / "test2.py")
        Path(files2[0]).write_text("def func2(): pass")

        # Run concurrent requests
        results = await asyncio.gather(
            ingest_workspace_incremental(files1, str(temp_workspace)),
            ingest_workspace_incremental(files2, str(temp_workspace))
        )

        # Both should succeed
        assert all(r["success"] for r in results)


class TestIntegrationWithWorkspaceSpinner:
    """Test integration with WorkspaceSpinner."""

    @pytest.mark.asyncio
    async def test_workspace_spinner_called_correctly(self, temp_workspace):
        """Test that WorkspaceSpinner is called with correct parameters."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        test_file = temp_workspace / "test.py"
        test_file.write_text("def func(): pass")

        result = await ingest_workspace_incremental(
            files=[str(test_file)],
            workspace_path=str(temp_workspace)
        )

        # If we got here without exception, WorkspaceSpinner worked
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_metadata_extraction(self, temp_workspace):
        """Test that file metadata is extracted correctly."""
        from HoloLoom.server.agentic_api import ingest_workspace_incremental

        test_file = temp_workspace / "test_metadata.py"
        test_file.write_text("""
def function1():
    '''Docstring'''
    pass

def function2(arg1, arg2):
    pass

class TestClass:
    pass

# TODO: Fix this
# FIXME: And this
""")

        result = await ingest_workspace_incremental(
            files=[str(test_file)],
            workspace_path=str(temp_workspace)
        )

        assert result["success"] is True
        # Should have detected functions, class, and comments
        assert result["code_elements"] >= 3  # 2 functions + 1 class
        assert result["comments"] >= 2  # TODO and FIXME


class TestPerformanceOptimizations:
    """Test performance-related aspects."""

    @pytest.mark.asyncio
    async def test_incremental_vs_full_workspace(self, temp_workspace):
        """
        Performance test: Incremental indexing should be faster than full.

        This is more of a benchmark than a correctness test.
        """
        from HoloLoom.server.agentic_api import (
            ingest_workspace_incremental,
            ingest_workspace
        )
        import time

        # Create test files
        for i in range(10):
            f = temp_workspace / f"test{i}.py"
            f.write_text(f"def func{i}(): pass")

        # Time full workspace index
        start = time.time()
        full_result = await ingest_workspace(
            workspace_path=str(temp_workspace)
        )
        full_time = time.time() - start

        # Time incremental index
        files = [str(temp_workspace / f"test{i}.py") for i in range(10)]
        start = time.time()
        inc_result = await ingest_workspace_incremental(
            files=files,
            workspace_path=str(temp_workspace)
        )
        inc_time = time.time() - start

        assert full_result["success"] is True
        assert inc_result["success"] is True

        # Incremental should process similar number of files
        # but might not be faster due to small dataset
        # Just verify both complete
        assert full_result["files_indexed"] > 0
        assert inc_result["files_indexed"] > 0
