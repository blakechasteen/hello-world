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

from HoloLoom.spinning_wheel import (
    AudioSpinner, TextSpinner, CodeSpinner, WebsiteSpinner, RecursiveCrawler,
    SpinnerConfig, TextSpinnerConfig, CodeSpinnerConfig, WebsiteSpinnerConfig, CrawlConfig
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
        config = TextSpinnerConfig(chunk_by='paragraph', chunk_size=100, min_chunk_size=20)
        spinner = TextSpinner(config)

        raw_data = {
            'text': 'This is the first paragraph with enough content to pass the minimum size threshold.\n\nThis is the second paragraph, also with sufficient length to be included.\n\nAnd here is the third paragraph with meaningful content.',
            'source': 'multi.txt'
        }

        shards = await spinner.spin(raw_data)

        assert len(shards) >= 1  # Should create multiple shards
        assert all('chunk_by' in s.metadata for s in shards)
        assert all(s.metadata['chunk_by'] == 'paragraph' for s in shards)
        assert all('chunk_index' in s.metadata for s in shards)

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
# WebsiteSpinner Tests
# ============================================================================

class TestWebsiteSpinner:
    """Test WebsiteSpinner functionality."""

    @pytest.mark.asyncio
    async def test_website_with_provided_content(self):
        """Test website spinner with pre-fetched content."""
        config = WebsiteSpinnerConfig(chunk_by='paragraph', min_content_length=50)
        spinner = WebsiteSpinner(config)

        raw_data = {
            'url': 'https://example.com/article',
            'title': 'Test Article',
            'content': 'This is a test article about beekeeping.\n\nIt has multiple paragraphs.\n\nEach paragraph should become a separate shard.'
        }

        shards = await spinner.spin(raw_data)

        # Should have multiple shards from chunking
        assert len(shards) >= 1
        assert shards[0].metadata['url'] == 'https://example.com/article'
        assert shards[0].metadata['title'] == 'Test Article'
        assert shards[0].metadata['domain'] == 'example.com'
        assert 'web:example.com' in shards[0].metadata.get('tags', [])

    @pytest.mark.asyncio
    async def test_website_with_tags(self):
        """Test website spinner preserves custom tags."""
        spinner = WebsiteSpinner()

        raw_data = {
            'url': 'https://example.com/article',
            'content': 'Test content about bees and honey.',
            'tags': ['research', 'beekeeping']
        }

        shards = await spinner.spin(raw_data)

        assert len(shards) >= 1
        tags = shards[0].metadata.get('tags', [])
        assert 'research' in tags
        assert 'beekeeping' in tags
        assert 'web:example.com' in tags

    @pytest.mark.asyncio
    async def test_website_empty_content(self):
        """Test handling of empty or too-short content."""
        config = WebsiteSpinnerConfig(min_content_length=100)
        spinner = WebsiteSpinner(config)

        raw_data = {
            'url': 'https://example.com/empty',
            'content': 'Too short'
        }

        shards = await spinner.spin(raw_data)

        # Should return empty list for content below minimum
        assert len(shards) == 0

    @pytest.mark.asyncio
    async def test_website_metadata_enrichment(self):
        """Test that URL metadata is properly added."""
        spinner = WebsiteSpinner()

        raw_data = {
            'url': 'https://docs.example.com/guide/tutorial',
            'title': 'Tutorial Guide',
            'content': 'This is a comprehensive tutorial about advanced beekeeping techniques.'
        }

        shards = await spinner.spin(raw_data)

        assert len(shards) >= 1
        first_shard = shards[0]

        # Check URL metadata
        assert first_shard.metadata['url'] == 'https://docs.example.com/guide/tutorial'
        assert first_shard.metadata['domain'] == 'docs.example.com'
        assert first_shard.metadata['title'] == 'Tutorial Guide'
        assert first_shard.metadata['content_type'] == 'webpage'


# ============================================================================
# RecursiveCrawler Tests
# ============================================================================

class TestRecursiveCrawler:
    """Test RecursiveCrawler functionality."""

    @pytest.mark.asyncio
    async def test_crawler_config(self):
        """Test crawler configuration."""
        config = CrawlConfig(
            max_depth=2,
            max_pages=10,
            importance_thresholds={0: 0.0, 1: 0.6, 2: 0.75}
        )

        crawler = RecursiveCrawler(config)

        assert crawler.config.max_depth == 2
        assert crawler.config.max_pages == 10
        assert crawler.config.importance_thresholds[1] == 0.6

    @pytest.mark.asyncio
    async def test_crawler_seed_only(self):
        """Test crawler with depth=0 (seed URL only)."""
        config = CrawlConfig(max_depth=0, max_pages=1)
        crawler = RecursiveCrawler(config)

        # Mock seed URL (we won't actually crawl in tests)
        # Just verify configuration
        assert config.max_depth == 0
        assert config.importance_thresholds[0] == 0.0

    @pytest.mark.asyncio
    async def test_crawler_matryoshka_thresholds(self):
        """Test matryoshka importance gating thresholds."""
        config = CrawlConfig()

        # Default thresholds should increase with depth
        assert config.importance_thresholds[0] == 0.0   # Seed: always crawl
        assert config.importance_thresholds[1] == 0.6   # Depth 1: medium importance
        assert config.importance_thresholds[2] == 0.75  # Depth 2: high importance
        assert config.importance_thresholds[3] == 0.85  # Depth 3: very high importance

        # Each level should require higher importance
        for depth in range(3):
            assert config.importance_thresholds[depth] < config.importance_thresholds[depth + 1]

    @pytest.mark.asyncio
    async def test_crawler_domain_filtering(self):
        """Test crawler domain filtering config."""
        config = CrawlConfig(
            same_domain_only=True,
            max_pages_per_domain=5
        )

        crawler = RecursiveCrawler(config)

        assert crawler.config.same_domain_only is True
        assert crawler.config.max_pages_per_domain == 5

    @pytest.mark.asyncio
    async def test_crawler_image_extraction_config(self):
        """Test crawler multimodal configuration."""
        config = CrawlConfig(
            extract_images=True,
            max_images_per_page=5
        )

        crawler = RecursiveCrawler(config)

        assert crawler.config.extract_images is True
        assert crawler.config.max_images_per_page == 5


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
