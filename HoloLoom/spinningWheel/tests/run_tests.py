#!/usr/bin/env python3
"""
Simple Test Runner for SpinningWheel
=====================================

Runs all spinner tests without requiring pytest.

Usage:
    python run_tests.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.spinningWheel import (
    AudioSpinner, TextSpinner, CodeSpinner,
    SpinnerConfig, TextSpinnerConfig, CodeSpinnerConfig
)


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record_pass(self, test_name):
        self.passed += 1
        print(f"  [PASS] {test_name}")

    def record_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"  [FAIL] {test_name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Tests: {total} total, {self.passed} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*60}")
        return self.failed == 0


results = TestResults()


# ============================================================================
# AudioSpinner Tests
# ============================================================================

async def test_audio_transcript_basic():
    """Test basic transcript ingestion."""
    spinner = AudioSpinner()

    raw_data = {
        'transcript': 'Today I inspected the hives and found them healthy.',
        'episode': 'test_session'
    }

    shards = await spinner.spin(raw_data)

    assert len(shards) == 1, f"Expected 1 shard, got {len(shards)}"
    assert shards[0].id == 'test_session_transcript'
    assert 'hives' in shards[0].text.lower()
    assert shards[0].episode == 'test_session'
    assert shards[0].metadata['type'] == 'transcript'


async def test_audio_multiple_types():
    """Test ingestion of transcript, summary, and tasks."""
    spinner = AudioSpinner()

    raw_data = {
        'transcript': 'Full transcript text here.',
        'summary': 'Summary of the session.',
        'tasks': ['Task 1', 'Task 2', 'Task 3'],
        'episode': 'multi_test'
    }

    shards = await spinner.spin(raw_data)

    assert len(shards) == 3, f"Expected 3 shards, got {len(shards)}"
    assert any(s.metadata['type'] == 'transcript' for s in shards)
    assert any(s.metadata['type'] == 'summary' for s in shards)
    assert any(s.metadata['type'] == 'tasks' for s in shards)


async def test_audio_empty_input():
    """Test handling of empty input."""
    spinner = AudioSpinner()

    raw_data = {}

    shards = await spinner.spin(raw_data)

    assert len(shards) == 0, f"Expected 0 shards, got {len(shards)}"


# ============================================================================
# TextSpinner Tests
# ============================================================================

async def test_text_single_shard():
    """Test single shard creation from text."""
    config = TextSpinnerConfig(chunk_by=None)
    spinner = TextSpinner(config)

    raw_data = {
        'text': 'This is a simple test document with some content.',
        'source': 'test.txt'
    }

    shards = await spinner.spin(raw_data)

    assert len(shards) == 1
    assert 'test document' in shards[0].text
    assert shards[0].metadata['source'] == 'test.txt'
    assert shards[0].metadata['format'] == 'text'


async def test_text_chunk_by_paragraph():
    """Test paragraph-based chunking."""
    config = TextSpinnerConfig(chunk_by='paragraph', chunk_size=100)
    spinner = TextSpinner(config)

    raw_data = {
        'text': 'Paragraph one.\n\nParagraph two.\n\nParagraph three.',
        'source': 'multi.txt'
    }

    shards = await spinner.spin(raw_data)

    assert len(shards) >= 1
    assert all('chunk_by' in s.metadata for s in shards)


async def test_text_entity_extraction():
    """Test basic entity extraction."""
    config = TextSpinnerConfig(extract_entities=True)
    spinner = TextSpinner(config)

    raw_data = {
        'text': 'John Smith visited New York and met Sarah Johnson.',
        'source': 'entities.txt'
    }

    shards = await spinner.spin(raw_data)

    assert len(shards[0].entities) > 0


async def test_text_missing_required_field():
    """Test error handling for missing required field."""
    spinner = TextSpinner()

    raw_data = {'source': 'test.txt'}  # Missing 'text'

    try:
        await spinner.spin(raw_data)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "'text' is required" in str(e)


# ============================================================================
# CodeSpinner Tests
# ============================================================================

async def test_code_file_python():
    """Test Python code file ingestion."""
    spinner = CodeSpinner()

    code = '''
import numpy as np

def hello():
    print("Hello, world!")

class MyClass:
    pass
'''

    raw_data = {
        'type': 'file',
        'path': 'test.py',
        'content': code
    }

    shards = await spinner.spin(raw_data)

    assert len(shards) == 1
    assert shards[0].metadata['language'] == 'python'
    assert 'hello' in shards[0].entities
    assert 'MyClass' in shards[0].entities
    assert 'numpy' in shards[0].metadata['imports']


async def test_code_chunk_by_function():
    """Test function-based code chunking."""
    config = CodeSpinnerConfig(chunk_by='function')
    spinner = CodeSpinner(config)

    code = '''
def func1():
    pass

def func2():
    pass

def func3():
    pass
'''

    raw_data = {
        'type': 'file',
        'path': 'funcs.py',
        'content': code
    }

    shards = await spinner.spin(raw_data)

    assert len(shards) == 3
    assert all('function_name' in s.metadata for s in shards)


async def test_code_language_detection():
    """Test automatic language detection."""
    spinner = CodeSpinner()

    test_cases = [
        ('test.py', 'python'),
        ('app.js', 'javascript'),
        ('Main.java', 'java'),
        ('main.go', 'go'),
        ('lib.rs', 'rust'),
    ]

    for filename, expected_lang in test_cases:
        raw_data = {
            'type': 'file',
            'path': filename,
            'content': '# test code'
        }

        shards = await spinner.spin(raw_data)
        assert shards[0].metadata['language'] == expected_lang, \
            f"Expected {expected_lang} for {filename}, got {shards[0].metadata['language']}"


async def test_code_git_diff():
    """Test git diff parsing."""
    spinner = CodeSpinner()

    diff = '''diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 import sys
+import os

 def main():
-    print("old")
+    print("new")
'''

    raw_data = {
        'type': 'diff',
        'diff': diff,
        'commit_sha': 'abc123',
        'message': 'Test commit'
    }

    shards = await spinner.spin(raw_data)

    assert len(shards) >= 1
    summary = [s for s in shards if s.metadata['type'] == 'diff_summary'][0]
    assert summary.metadata['commit_sha'] == 'abc123'
    assert summary.metadata['insertions'] > 0


async def test_code_repo_structure():
    """Test repository structure mapping."""
    spinner = CodeSpinner()

    files = ['src/main.py', 'src/utils.py', 'tests/test.py']

    raw_data = {
        'type': 'repo',
        'root_path': '/project',
        'files': files
    }

    shards = await spinner.spin(raw_data)

    assert len(shards) == 1
    assert shards[0].metadata['type'] == 'repo_structure'
    assert shards[0].metadata['file_count'] == 3
    assert 'python' in shards[0].metadata['languages']


# ============================================================================
# Test Runner
# ============================================================================

async def run_test_suite():
    """Run all tests."""

    test_functions = [
        # AudioSpinner tests
        ("AudioSpinner: basic transcript", test_audio_transcript_basic),
        ("AudioSpinner: multiple types", test_audio_multiple_types),
        ("AudioSpinner: empty input", test_audio_empty_input),

        # TextSpinner tests
        ("TextSpinner: single shard", test_text_single_shard),
        ("TextSpinner: paragraph chunking", test_text_chunk_by_paragraph),
        ("TextSpinner: entity extraction", test_text_entity_extraction),
        ("TextSpinner: missing field error", test_text_missing_required_field),

        # CodeSpinner tests
        ("CodeSpinner: Python file", test_code_file_python),
        ("CodeSpinner: function chunking", test_code_chunk_by_function),
        ("CodeSpinner: language detection", test_code_language_detection),
        ("CodeSpinner: git diff", test_code_git_diff),
        ("CodeSpinner: repo structure", test_code_repo_structure),
    ]

    print("\n" + "=" * 60)
    print("Running SpinningWheel Unit Tests")
    print("=" * 60 + "\n")

    for test_name, test_func in test_functions:
        try:
            await test_func()
            results.record_pass(test_name)
        except AssertionError as e:
            results.record_fail(test_name, str(e))
        except Exception as e:
            results.record_fail(test_name, f"Error: {type(e).__name__}: {e}")

    return results.summary()


if __name__ == '__main__':
    success = asyncio.run(run_test_suite())
    sys.exit(0 if success else 1)
