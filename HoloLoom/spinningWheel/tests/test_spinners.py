#!/usr/bin/env python3
"""
Unit Tests for SpinningWheel Spinners
======================================

Tests all spinner implementations:
- AudioSpinner
- TextSpinner
- YouTubeSpinner
- CodeSpinner
- WebsiteSpinner

Run with:
    pytest test_spinners.py -v
    or
    python -m pytest test_spinners.py -v
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.spinningWheel import (
    AudioSpinner, TextSpinner, CodeSpinner,
    SpinnerConfig, TextSpinnerConfig, CodeSpinnerConfig
)


# ============================================================================
# AudioSpinner Tests
# ============================================================================

class TestAudioSpinner:
    """Test AudioSpinner functionality."""

    @pytest.mark.asyncio
    async def test_audio_transcript_basic(self):
        """Test basic transcript ingestion."""
        spinner = AudioSpinner()

        raw_data = {
            'transcript': 'Today I inspected the hives and found them healthy.',
            'episode': 'test_session'
        }

        shards = await spinner.spin(raw_data)

        assert len(shards) == 1
        assert shards[0].id == 'test_session_transcript'
        assert 'hives' in shards[0].text.lower()
        assert shards[0].episode == 'test_session'
        assert shards[0].metadata['type'] == 'transcript'

    @pytest.mark.asyncio
    async def test_audio_multiple_types(self):
        """Test ingestion of transcript, summary, and tasks."""
        spinner = AudioSpinner()

        raw_data = {
            'transcript': 'Full transcript text here.',
            'summary': 'Summary of the session.',
            'tasks': ['Task 1', 'Task 2', 'Task 3'],
            'episode': 'multi_test'
        }

        shards = await spinner.spin(raw_data)

        assert len(shards) == 3
        assert any(s.metadata['type'] == 'transcript' for s in shards)
        assert any(s.metadata['type'] == 'summary' for s in shards)
        assert any(s.metadata['type'] == 'tasks' for s in shards)

    @pytest.mark.asyncio
    async def test_audio_empty_input(self):
        """Test handling of empty input."""
        spinner = AudioSpinner()

        raw_data = {}

        shards = await spinner.spin(raw_data)

        assert len(shards) == 0


# ============================================================================
# TextSpinner Tests
# ============================================================================

class TestTextSpinner:
    """Test TextSpinner functionality."""

    @pytest.mark.asyncio
    async def test_text_single_shard(self):
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

    @pytest.mark.asyncio
    async def test_text_chunk_by_paragraph(self):
        """Test paragraph-based chunking."""
        config = TextSpinnerConfig(chunk_by='paragraph', chunk_size=100)
        spinner = TextSpinner(config)

        raw_data = {
            'text': 'Paragraph one.\n\nParagraph two.\n\nParagraph three.',
            'source': 'multi.txt'
        }

        shards = await spinner.spin(raw_data)

        assert len(shards) >= 1  # May combine paragraphs if under chunk_size
        assert all('chunk_by' in s.metadata for s in shards)
        assert all(s.metadata['chunk_by'] == 'paragraph' for s in shards)

    @pytest.mark.asyncio
    async def test_text_chunk_by_sentence(self):
        """Test sentence-based chunking."""
        config = TextSpinnerConfig(chunk_by='sentence', chunk_size=50)
        spinner = TextSpinner(config)

        raw_data = {
            'text': 'First sentence. Second sentence. Third sentence here.',
            'source': 'sentences.txt'
        }

        shards = await spinner.spin(raw_data)

        assert len(shards) >= 1
        assert all(s.metadata['chunk_by'] == 'sentence' for s in shards)

    @pytest.mark.asyncio
    async def test_text_entity_extraction(self):
        """Test basic entity extraction."""
        config = TextSpinnerConfig(extract_entities=True)
        spinner = TextSpinner(config)

        raw_data = {
            'text': 'John Smith visited New York and met Sarah Johnson.',
            'source': 'entities.txt'
        }

        shards = await spinner.spin(raw_data)

        assert len(shards[0].entities) > 0
        # Should extract capitalized words as entities

    @pytest.mark.asyncio
    async def test_text_missing_required_field(self):
        """Test error handling for missing required field."""
        spinner = TextSpinner()

        raw_data = {'source': 'test.txt'}  # Missing 'text'

        with pytest.raises(ValueError, match="'text' is required"):
            await spinner.spin(raw_data)


# ============================================================================
# CodeSpinner Tests
# ============================================================================

class TestCodeSpinner:
    """Test CodeSpinner functionality."""

    @pytest.mark.asyncio
    async def test_code_file_python(self):
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

    @pytest.mark.asyncio
    async def test_code_chunk_by_function(self):
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

    @pytest.mark.asyncio
    async def test_code_language_detection(self):
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
            assert shards[0].metadata['language'] == expected_lang

    @pytest.mark.asyncio
    async def test_code_git_diff(self):
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

    @pytest.mark.asyncio
    async def test_code_repo_structure(self):
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

    @pytest.mark.asyncio
    async def test_code_import_extraction(self):
        """Test import statement extraction."""
        config = CodeSpinnerConfig(extract_imports=True)
        spinner = CodeSpinner(config)

        code = '''
import os
import sys
from pathlib import Path
from typing import List, Dict
'''

        raw_data = {
            'type': 'file',
            'path': 'imports.py',
            'content': code
        }

        shards = await spinner.spin(raw_data)

        imports = shards[0].metadata['imports']
        assert 'os' in imports
        assert 'sys' in imports
        assert 'pathlib' in imports
        assert 'typing' in imports


# ============================================================================
# Integration Tests
# ============================================================================

class TestSpinnerIntegration:
    """Test spinner integration and error handling."""

    @pytest.mark.asyncio
    async def test_all_spinners_produce_valid_shards(self):
        """Test that all spinners produce valid MemoryShards."""
        # Audio
        audio = AudioSpinner()
        audio_shards = await audio.spin({'transcript': 'test'})
        assert all(hasattr(s, 'id') and hasattr(s, 'text') for s in audio_shards)

        # Text
        text = TextSpinner()
        text_shards = await text.spin({'text': 'test'})
        assert all(hasattr(s, 'id') and hasattr(s, 'text') for s in text_shards)

        # Code
        code = CodeSpinner()
        code_shards = await code.spin({'type': 'file', 'path': 't.py', 'content': 'pass'})
        assert all(hasattr(s, 'id') and hasattr(s, 'text') for s in code_shards)

    @pytest.mark.asyncio
    async def test_empty_content_handling(self):
        """Test handling of empty content."""
        # Empty transcript
        audio = AudioSpinner()
        audio_shards = await audio.spin({})
        assert len(audio_shards) == 0

        # Empty text should raise error
        text = TextSpinner()
        with pytest.raises(ValueError):
            await text.spin({'source': 'test.txt'})

    @pytest.mark.asyncio
    async def test_metadata_preservation(self):
        """Test that custom metadata is preserved."""
        spinner = TextSpinner()

        raw_data = {
            'text': 'test content',
            'metadata': {
                'author': 'test_user',
                'tags': ['important', 'test']
            }
        }

        shards = await spinner.spin(raw_data)

        assert shards[0].metadata['author'] == 'test_user'
        assert shards[0].metadata['tags'] == ['important', 'test']


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
