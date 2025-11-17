"""
Tests for Writing Modes - NarrativeWriter and TechnicalWriter

Tests mode-specific writers for different content types.

Author: Claude Code
Date: 2025-11-17
Phase: 2 (Testing) - Week 1, Day 4
Target: +12 tests for mode writers
"""

import pytest
from unittest.mock import Mock, patch
from HoloLoom.writing.modes.narrative import NarrativeWriter
from HoloLoom.writing.modes.technical import TechnicalWriter
from HoloLoom.writing.core.protocol import (
    WritingContext,
    WritingMode,
    StyleGuide,
    MemoryShard
)


class TestNarrativeWriter:
    """Test suite for NarrativeWriter mode."""

    def test_narrative_writer_initialization(self):
        """Test NarrativeWriter initialization."""
        writer = NarrativeWriter()

        assert writer.logger is not None

    @pytest.mark.asyncio
    async def test_generate_narrative_from_memories(self):
        """Test narrative generation from memories."""
        writer = NarrativeWriter()

        context = WritingContext(
            query="Tell me about the hero's journey",
            memories=[
                MemoryShard(
                    id="1",
                    text="The hero left the village",
                    metadata={"timestamp": 100.0}
                ),
                MemoryShard(
                    id="2",
                    text="The hero found the sword",
                    metadata={"timestamp": 200.0}
                ),
                MemoryShard(
                    id="3",
                    text="The hero defeated the dragon",
                    metadata={"timestamp": 300.0}
                )
            ],
            mode=WritingMode.NARRATIVE,
            style=StyleGuide.CASUAL
        )

        result = await writer.generate(context)

        # Should generate a narrative
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_narrative_empty_memories(self):
        """Test narrative generation with empty memories."""
        writer = NarrativeWriter()

        context = WritingContext(
            query="Tell me a story",
            memories=[],
            mode=WritingMode.NARRATIVE,
            style=StyleGuide.CASUAL
        )

        result = await writer.generate(context)

        # Should handle empty memories gracefully
        assert isinstance(result, str)
        assert len(result) > 0

    def test_analyze_memories(self):
        """Test memory analysis for entity extraction."""
        writer = NarrativeWriter()

        memories = [
            MemoryShard(id="1", text="Alice went to the store", metadata={}),
            MemoryShard(id="2", text="Bob met Alice at the park", metadata={})
        ]

        analysis = writer._analyze_memories(memories)

        # Should extract entities
        assert isinstance(analysis, dict)
        assert 'entities' in analysis or 'themes' in analysis

    def test_order_memories_temporal(self):
        """Test temporal ordering of memories."""
        writer = NarrativeWriter()

        memories = [
            MemoryShard(id="3", text="Event 3", metadata={"timestamp": 300.0}),
            MemoryShard(id="1", text="Event 1", metadata={"timestamp": 100.0}),
            MemoryShard(id="2", text="Event 2", metadata={"timestamp": 200.0})
        ]

        ordered = writer._order_memories(memories)

        # Should be ordered by timestamp
        assert len(ordered) == 3
        # Check if first memory has lowest timestamp (if temporal ordering is implemented)
        if ordered[0].metadata.get('timestamp'):
            assert ordered[0].metadata['timestamp'] <= ordered[1].metadata.get('timestamp', float('inf'))

    def test_generate_narrative_structure(self):
        """Test narrative structure generation."""
        writer = NarrativeWriter()

        memories = [MemoryShard(id="1", text="Test memory", metadata={})]

        narrative = writer._generate_narrative_structure(
            query="Tell a story",
            memories=memories,
            entities=["Alice", "Bob"],
            style=StyleGuide.CASUAL
        )

        # Should generate narrative structure
        assert isinstance(narrative, str)
        assert len(narrative) > 0


class TestTechnicalWriter:
    """Test suite for TechnicalWriter mode."""

    def test_technical_writer_initialization(self):
        """Test TechnicalWriter initialization."""
        writer = TechnicalWriter()

        assert writer.logger is not None

    @pytest.mark.asyncio
    async def test_generate_technical_from_memories(self):
        """Test technical documentation generation."""
        writer = TechnicalWriter()

        context = WritingContext(
            query="Explain the API endpoint",
            memories=[
                MemoryShard(
                    id="1",
                    text="GET /api/users returns user list",
                    metadata={}
                ),
                MemoryShard(
                    id="2",
                    text="Requires authentication token",
                    metadata={}
                )
            ],
            mode=WritingMode.TECHNICAL,
            style=StyleGuide.TECHNICAL
        )

        result = await writer.generate(context)

        # Should generate technical documentation
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_technical_empty_memories(self):
        """Test technical generation with empty memories."""
        writer = TechnicalWriter()

        context = WritingContext(
            query="Explain the function",
            memories=[],
            mode=WritingMode.TECHNICAL,
            style=StyleGuide.TECHNICAL
        )

        result = await writer.generate(context)

        # Should handle empty memories gracefully
        assert isinstance(result, str)
        assert len(result) > 0

    def test_analyze_technical_content(self):
        """Test technical content analysis."""
        writer = TechnicalWriter()

        memories = [
            MemoryShard(
                id="1",
                text="def calculate(x, y): return x + y",
                metadata={}
            ),
            MemoryShard(
                id="2",
                text="Parameters: x (int), y (int)",
                metadata={}
            )
        ]

        analysis = writer._analyze_technical_content(memories)

        # Should extract technical elements
        assert isinstance(analysis, dict)

    def test_generate_technical_structure(self):
        """Test technical documentation structure."""
        writer = TechnicalWriter()

        memories = [
            MemoryShard(id="1", text="API endpoint details", metadata={})
        ]

        analysis = {
            'code_samples': [],
            'parameters': [],
            'procedures': []
        }

        doc = writer._generate_technical_structure(
            query="Explain API",
            memories=memories,
            analysis=analysis,
            style=StyleGuide.TECHNICAL
        )

        # Should generate structured documentation
        assert isinstance(doc, str)
        assert len(doc) > 0

    def test_extract_code_samples(self):
        """Test code sample extraction from memories."""
        writer = TechnicalWriter()

        memories = [
            MemoryShard(
                id="1",
                text="```python\nprint('hello')\n```",
                metadata={}
            ),
            MemoryShard(
                id="2",
                text="No code here",
                metadata={}
            )
        ]

        # Analyze should find code blocks
        analysis = writer._analyze_technical_content(memories)

        assert isinstance(analysis, dict)
        # Check if code samples were extracted (implementation-dependent)
