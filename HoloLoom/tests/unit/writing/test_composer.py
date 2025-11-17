"""
Tests for Writing Core - Composer Class

Tests the Composer class for multi-pass content refinement.

Author: Claude Code
Date: 2025-11-17
Phase: 2 (Testing) - Week 1, Day 3
Target: +8 tests for Composer class
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from HoloLoom.writing.core.composer import Composer
from HoloLoom.writing.core.protocol import (
    WritingContext,
    WritingResult,
    WritingMode,
    StyleGuide,
    RefinementStrategy,
    RefinementPass,
    MemoryShard
)


class TestComposerInitialization:
    """Test suite for Composer initialization."""

    def test_composer_initialization_empty(self):
        """Test Composer initialization with no refiners."""
        composer = Composer()

        assert composer.refiners == {}
        assert composer.logger is not None

    def test_composer_initialization_with_refiners(self):
        """Test Composer initialization with custom refiners."""
        mock_elegance_refiner = Mock()
        mock_verify_refiner = Mock()

        composer = Composer(
            refiners={
                RefinementStrategy.ELEGANCE: mock_elegance_refiner,
                RefinementStrategy.VERIFY: mock_verify_refiner
            }
        )

        assert RefinementStrategy.ELEGANCE in composer.refiners
        assert RefinementStrategy.VERIFY in composer.refiners


class TestSinglePassRefinement:
    """Test suite for single-pass refinement."""

    @pytest.mark.asyncio
    async def test_compose_single_pass(self):
        """Test composition with single refinement pass."""
        # Mock refiner
        mock_refiner = Mock()
        mock_pass = RefinementPass(
            pass_number=1,
            strategy="elegance",
            focus="clarity",
            input_text="Initial draft",
            output_text="Refined draft",
            quality_score=0.85,
            improvements=["Improved clarity"],
            duration_ms=50.0
        )
        mock_refiner.refine = AsyncMock(return_value=mock_pass)

        composer = Composer(refiners={RefinementStrategy.ELEGANCE: mock_refiner})

        context = WritingContext(
            query="Test query",
            memories=[MemoryShard(id="1", text="Memory", metadata={})],
            mode=WritingMode.NARRATIVE,
            style=StyleGuide.TUFTE
        )

        result = await composer.compose(
            initial_draft="Initial draft",
            context=context,
            strategy=RefinementStrategy.ELEGANCE,
            max_passes=1,
            quality_threshold=0.9
        )

        # Should have 1 refinement pass
        assert result.total_passes == 1
        assert result.content == "Refined draft"
        assert result.quality_score >= 0.0


class TestMultiPassRefinement:
    """Test suite for multi-pass refinement."""

    @pytest.mark.asyncio
    async def test_compose_multiple_passes(self):
        """Test composition with multiple refinement passes."""
        # Mock refiner that improves quality each pass
        mock_refiner = Mock()

        pass_counter = [0]  # Use list to allow modification in nested function

        async def refine_mock(text, context, pass_number):
            quality = 0.6 + (pass_number * 0.15)  # Quality improves each pass
            return RefinementPass(
                pass_number=pass_number,
                strategy="elegance",
                focus=f"pass_{pass_number}",
                input_text=text,
                output_text=f"Refined pass {pass_number}",
                quality_score=quality,
                improvements=[f"Improvement {pass_number}"],
                duration_ms=50.0
            )

        mock_refiner.refine = refine_mock

        composer = Composer(refiners={RefinementStrategy.ELEGANCE: mock_refiner})

        context = WritingContext(
            query="Test",
            memories=[],
            mode=WritingMode.NARRATIVE,
            style=StyleGuide.TUFTE
        )

        # Mock score_quality to show improvement each pass
        def mock_quality_score(text, ctx):
            pass_counter[0] += 1
            score = 0.6 + (pass_counter[0] * 0.1)  # Increases each call
            return (score, {'clarity': score, 'completeness': score})

        with patch.object(composer, 'score_quality', side_effect=mock_quality_score):
            result = await composer.compose(
                initial_draft="Draft",
                context=context,
                strategy=RefinementStrategy.ELEGANCE,
                max_passes=3,
                quality_threshold=0.95  # High threshold to ensure all 3 passes run
            )

            # Should have 3 passes
            assert result.total_passes == 3
            # Quality should improve
            trajectory = result.quality_trajectory
            assert len(trajectory) == 3
            # Later passes should have higher quality
            assert trajectory[2] > trajectory[0]

    @pytest.mark.asyncio
    async def test_compose_early_termination(self):
        """Test composition terminates early when quality threshold reached."""
        # Mock refiner that reaches threshold on pass 2
        mock_refiner = Mock()

        async def refine_mock(text, context, pass_number):
            quality = 0.92 if pass_number >= 2 else 0.7
            return RefinementPass(
                pass_number=pass_number,
                strategy="verify",
                focus="accuracy",
                input_text=text,
                output_text=f"Refined {pass_number}",
                quality_score=quality,
                improvements=["Improvement"],
                duration_ms=50.0
            )

        mock_refiner.refine = refine_mock

        composer = Composer(refiners={RefinementStrategy.VERIFY: mock_refiner})

        context = WritingContext(
            query="Test",
            memories=[],
            mode=WritingMode.TECHNICAL,
            style=StyleGuide.TECHNICAL
        )

        result = await composer.compose(
            initial_draft="Draft",
            context=context,
            strategy=RefinementStrategy.VERIFY,
            max_passes=5,
            quality_threshold=0.9
        )

        # Should terminate after pass 2 (reaches 0.92 quality)
        assert result.total_passes <= 3


class TestQualityScoring:
    """Test suite for quality scoring."""

    def test_score_quality_basic(self):
        """Test basic quality scoring."""
        composer = Composer()

        context = WritingContext(
            query="Test",
            memories=[],
            mode=WritingMode.NARRATIVE,
            style=StyleGuide.CASUAL
        )

        score, dimensions = composer.score_quality("Test content", context)

        # Should return score and dimensions
        assert 0.0 <= score <= 1.0
        assert isinstance(dimensions, dict)

    def test_score_quality_empty_text(self):
        """Test quality scoring with empty text."""
        composer = Composer()

        context = WritingContext(
            query="Test",
            memories=[],
            mode=WritingMode.NARRATIVE,
            style=StyleGuide.CASUAL
        )

        score, dimensions = composer.score_quality("", context)

        # Empty text should have low quality
        assert score < 0.5


class TestStrategySelection:
    """Test suite for refinement strategy handling."""

    @pytest.mark.asyncio
    async def test_compose_with_missing_refiner(self):
        """Test composition falls back to basic refiner when custom refiner missing."""
        composer = Composer(refiners={})  # No refiners

        context = WritingContext(
            query="Test",
            memories=[],
            mode=WritingMode.NARRATIVE,
            style=StyleGuide.CASUAL
        )

        # Should use built-in basic refiner
        with patch.object(composer, '_get_basic_refiner') as mock_basic:
            mock_refiner = Mock()
            mock_pass = RefinementPass(
                pass_number=1,
                strategy="elegance",
                focus="clarity",
                input_text="Draft",
                output_text="Refined",
                quality_score=0.8,
                improvements=[],
                duration_ms=50.0
            )
            mock_refiner.refine = AsyncMock(return_value=mock_pass)
            mock_basic.return_value = mock_refiner

            result = await composer.compose(
                initial_draft="Draft",
                context=context,
                strategy=RefinementStrategy.ELEGANCE,
                max_passes=1
            )

            # Should call _get_basic_refiner
            mock_basic.assert_called_once_with(RefinementStrategy.ELEGANCE)
            assert result.total_passes == 1


class TestResultSummary:
    """Test suite for result summary generation."""

    def test_writing_result_summary(self):
        """Test WritingResult.summary() method."""
        result = WritingResult(
            content="Final content",
            mode=WritingMode.NARRATIVE,
            style=StyleGuide.TUFTE,
            quality_score=0.92,
            confidence=0.88,
            refinement_passes=[
                RefinementPass(
                    pass_number=1,
                    strategy="elegance",
                    focus="clarity",
                    input_text="Draft",
                    output_text="Pass 1",
                    quality_score=0.75,
                    improvements=["clarity"],
                    duration_ms=50.0
                ),
                RefinementPass(
                    pass_number=2,
                    strategy="elegance",
                    focus="simplicity",
                    input_text="Pass 1",
                    output_text="Pass 2",
                    quality_score=0.92,
                    improvements=["simplicity"],
                    duration_ms=50.0
                )
            ],
            metadata={}
        )

        summary = result.summary()

        # Should contain key information
        assert "narrative" in summary.lower()
        assert "2" in summary  # 2 passes
        assert "0.75" in summary or "0.92" in summary  # Quality scores
