"""
Tests for Writing Core - Writer Class

Tests the main Writer class for content generation from memory.

Author: Claude Code
Date: 2025-11-17
Phase: 2 (Testing) - Week 1, Day 3
Target: +10 tests for Writer class
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from HoloLoom.writing.core.writer import Writer
from HoloLoom.writing.core.protocol import (
    WritingContext,
    WritingResult,
    WritingMode,
    StyleGuide,
    RefinementStrategy,
    RefinementPass,
    MemoryShard
)


class TestWriterInitialization:
    """Test suite for Writer initialization."""

    def test_writer_initialization_empty(self):
        """Test Writer initialization with no arguments."""
        writer = Writer()

        assert writer.mode_writers == {}
        assert writer.composer is None
        assert writer.logger is not None

    def test_writer_initialization_with_composers(self):
        """Test Writer initialization with mode writers and composer."""
        mock_narrative_writer = Mock()
        mock_composer = Mock()

        writer = Writer(
            mode_writers={WritingMode.NARRATIVE: mock_narrative_writer},
            composer=mock_composer
        )

        assert WritingMode.NARRATIVE in writer.mode_writers
        assert writer.mode_writers[WritingMode.NARRATIVE] == mock_narrative_writer
        assert writer.composer == mock_composer


class TestModeDetection:
    """Test suite for WritingMode auto-detection."""

    def test_detect_mode_narrative_query(self):
        """Test mode detection for narrative-style query."""
        writer = Writer()
        memories = [
            MemoryShard(id="1", text="Once upon a time", metadata={})
        ]

        mode = writer.detect_mode("Tell me a story about dragons", memories)

        # Should detect narrative mode
        assert mode in [WritingMode.NARRATIVE, WritingMode.CREATIVE]

    def test_detect_mode_technical_query(self):
        """Test mode detection for technical query."""
        writer = Writer()
        memories = [
            MemoryShard(id="1", text="API documentation: GET /users", metadata={})
        ]

        mode = writer.detect_mode("Explain the API endpoint", memories)

        # Should detect technical mode
        assert mode in [WritingMode.TECHNICAL, WritingMode.CODE_DOC]


class TestStyleDetection:
    """Test suite for StyleGuide auto-detection."""

    def test_detect_style_from_mode(self):
        """Test style detection based on mode."""
        writer = Writer()
        memories = []

        # Technical mode should suggest technical style
        style = writer.detect_style(
            "Explain the function",
            memories,
            WritingMode.TECHNICAL
        )

        assert style in [StyleGuide.TECHNICAL, StyleGuide.ACADEMIC]

    def test_detect_style_creative_mode(self):
        """Test style detection for creative mode."""
        writer = Writer()
        memories = []

        style = writer.detect_style(
            "Write a poem",
            memories,
            WritingMode.CREATIVE
        )

        assert style == StyleGuide.CREATIVE


class TestWritingWithoutRefinement:
    """Test suite for basic writing without refinement."""

    @pytest.mark.asyncio
    async def test_write_without_refinement(self):
        """Test writing content without refinement."""
        # Mock mode writer
        mock_mode_writer = Mock()
        mock_mode_writer.generate = AsyncMock(return_value="Generated content")

        writer = Writer(
            mode_writers={WritingMode.NARRATIVE: mock_mode_writer},
            composer=None  # No composer = no refinement
        )

        context = WritingContext(
            query="Test query",
            memories=[MemoryShard(id="1", text="Test memory", metadata={})],
            mode=WritingMode.NARRATIVE,
            style=StyleGuide.CASUAL
        )

        result = await writer.write(context, refine=False)

        assert result.content is not None
        assert result.mode == WritingMode.NARRATIVE
        assert result.style == StyleGuide.CASUAL
        assert 0.0 <= result.quality_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_write_auto_mode_detection(self):
        """Test writing with AUTO mode triggers mode detection."""
        writer = Writer()

        context = WritingContext(
            query="Tell me about Thompson Sampling",
            memories=[MemoryShard(id="1", text="Statistics", metadata={})],
            mode=WritingMode.AUTO,  # Should auto-detect
            style=StyleGuide.AUTO
        )

        # Patch the _generate_draft method to avoid actual generation
        with patch.object(writer, '_generate_draft', new=AsyncMock(return_value="Draft")):
            with patch.object(writer, '_score_quality', return_value=(0.8, {})):
                result = await writer.write(context, refine=False)

                # Mode should have been detected
                assert result.mode != WritingMode.AUTO
                assert result.style != StyleGuide.AUTO


class TestWritingWithRefinement:
    """Test suite for writing with multi-pass refinement."""

    @pytest.mark.asyncio
    async def test_write_with_refinement(self):
        """Test writing with refinement enabled."""
        # Mock composer
        mock_composer = Mock()
        mock_result = WritingResult(
            content="Refined content",
            mode=WritingMode.NARRATIVE,
            style=StyleGuide.TUFTE,
            quality_score=0.95,
            confidence=0.92,
            refinement_passes=[
                RefinementPass(
                    pass_number=1,
                    strategy="elegance",
                    focus="clarity",
                    input_text="Draft",
                    output_text="Refined content",
                    quality_score=0.95,
                    improvements=["Improved clarity"],
                    duration_ms=50.0
                )
            ],
            metadata={}
        )
        mock_composer.compose = AsyncMock(return_value=mock_result)

        writer = Writer(composer=mock_composer)

        context = WritingContext(
            query="Test query",
            memories=[MemoryShard(id="1", text="Memory", metadata={})],
            mode=WritingMode.NARRATIVE,
            style=StyleGuide.TUFTE
        )

        with patch.object(writer, '_generate_draft', new=AsyncMock(return_value="Draft")):
            result = await writer.write(context, refine=True)

            # Should call composer
            mock_composer.compose.assert_called_once()

            # Should return refined result
            assert result.content == "Refined content"
            assert result.quality_score == 0.95
            assert len(result.refinement_passes) > 0


class TestQualityScoring:
    """Test suite for quality scoring."""

    def test_score_quality_basic(self):
        """Test basic quality scoring."""
        writer = Writer()

        context = WritingContext(
            query="Test query",
            memories=[MemoryShard(id="1", text="Memory", metadata={})],
            mode=WritingMode.NARRATIVE,
            style=StyleGuide.CASUAL
        )

        score, details = writer._score_quality("Test content", context)

        # Should return a score between 0 and 1
        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)


class TestRefinementStrategy:
    """Test suite for refinement strategy selection."""

    def test_select_refinement_strategy_narrative(self):
        """Test strategy selection for narrative mode."""
        writer = Writer()

        strategy = writer._select_refinement_strategy(WritingMode.NARRATIVE)

        # Narrative mode should use ELEGANCE strategy
        assert strategy in [RefinementStrategy.ELEGANCE, RefinementStrategy.COHERENCE]

    def test_select_refinement_strategy_technical(self):
        """Test strategy selection for technical mode."""
        writer = Writer()

        strategy = writer._select_refinement_strategy(WritingMode.TECHNICAL)

        # Technical mode should use VERIFY strategy
        assert strategy in [RefinementStrategy.VERIFY, RefinementStrategy.COHERENCE]
