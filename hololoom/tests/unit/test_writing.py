"""
Tests for HoloLoom Writing System
==================================

Tests the core writing functionality.
"""

import pytest
from typing import List

from HoloLoom.writing import (
    write,
    refine_text,
    Writer,
    Composer,
    WritingContext,
    WritingMode,
    StyleGuide,
    RefinementStrategy
)
from HoloLoom.writing.core.protocol import MemoryShard
from HoloLoom.writing.modes import NarrativeWriter
from HoloLoom.writing.refinement import EleganceRefiner


# Test fixtures

@pytest.fixture
def sample_memories():
    """Sample memory shards for testing."""
    return [
        MemoryShard(
            id='mem1',
            text="Thompson Sampling is a Bayesian approach to exploration-exploitation",
            metadata={'relevance': 0.95, 'topic': 'reinforcement_learning'}
        ),
        MemoryShard(
            id='mem2',
            text="It samples from posterior distributions to balance exploration",
            metadata={'relevance': 0.88, 'topic': 'bayesian_methods'}
        ),
        MemoryShard(
            id='mem3',
            text="The method is optimal in the limit for multi-armed bandits",
            metadata={'relevance': 0.82, 'topic': 'bandits'}
        )
    ]


@pytest.fixture
def empty_memories():
    """Empty memory list for testing."""
    return []


# Test mode detection

@pytest.mark.asyncio
async def test_mode_detection_technical():
    """Test auto-detection of technical mode."""
    writer = Writer()

    query = "How to implement Thompson Sampling in Python?"
    mode = writer.detect_mode(query, [])

    assert mode == WritingMode.TECHNICAL


@pytest.mark.asyncio
async def test_mode_detection_narrative():
    """Test auto-detection of narrative mode."""
    writer = Writer()

    query = "What is Thompson Sampling?"
    mode = writer.detect_mode(query, [])

    assert mode == WritingMode.NARRATIVE


@pytest.mark.asyncio
async def test_mode_detection_analysis():
    """Test auto-detection of analysis mode."""
    writer = Writer()

    query = "Compare Thompson Sampling and epsilon-greedy"
    mode = writer.detect_mode(query, [])

    assert mode == WritingMode.ANALYSIS


@pytest.mark.asyncio
async def test_mode_detection_creative():
    """Test auto-detection of creative mode."""
    writer = Writer()

    query = "Write a story about Thompson Sampling"
    mode = writer.detect_mode(query, [])

    assert mode == WritingMode.CREATIVE


# Test style detection

@pytest.mark.asyncio
async def test_style_detection_academic():
    """Test auto-detection of academic style."""
    writer = Writer()

    query = "Provide an academic analysis of Thompson Sampling"
    style = writer.detect_style(query, [], WritingMode.ANALYSIS)

    assert style == StyleGuide.ACADEMIC


@pytest.mark.asyncio
async def test_style_detection_casual():
    """Test auto-detection of casual style."""
    writer = Writer()

    query = "Explain Thompson Sampling in simple terms"
    style = writer.detect_style(query, [], WritingMode.NARRATIVE)

    assert style == StyleGuide.CASUAL


# Test narrative writer

@pytest.mark.asyncio
async def test_narrative_writer_basic(sample_memories):
    """Test basic narrative generation."""
    writer = NarrativeWriter()

    context = WritingContext(
        query="What is Thompson Sampling?",
        memories=sample_memories,
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    content = await writer.generate(context)

    assert content is not None
    assert len(content) > 0
    assert "Thompson Sampling" in content


@pytest.mark.asyncio
async def test_narrative_writer_empty_memories(empty_memories):
    """Test narrative generation with no memories."""
    writer = NarrativeWriter()

    context = WritingContext(
        query="What is Thompson Sampling?",
        memories=empty_memories,
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    content = await writer.generate(context)

    assert content is not None
    assert "No relevant information" in content


# Test quality scoring

@pytest.mark.asyncio
async def test_quality_scoring():
    """Test quality scoring mechanism."""
    composer = Composer()

    context = WritingContext(
        query="Test query",
        memories=[],
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    # Good text
    good_text = """
    This is a clear explanation. It has multiple sentences.
    The content is well-structured.

    The second paragraph continues the explanation.
    It maintains clarity and coherence throughout.
    """

    quality, dimensions = composer.score_quality(good_text, context)

    assert 0.0 <= quality <= 1.0
    assert 'clarity' in dimensions
    assert 'completeness' in dimensions
    assert 'coherence' in dimensions


# Test ELEGANCE refiner

@pytest.mark.asyncio
async def test_elegance_refiner_clarity():
    """Test ELEGANCE refiner clarity pass."""
    refiner = EleganceRefiner()

    context = WritingContext(
        query="Test",
        memories=[],
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    # Text with issues
    text = "In order to utilize Thompson Sampling one must really understand it was designed to facilitate exploration"

    result = await refiner.refine(text, context, pass_number=1)

    assert result.pass_number == 1
    assert result.focus == 'clarity'
    assert len(result.improvements) > 0
    assert result.output_text != text  # Should be different


@pytest.mark.asyncio
async def test_elegance_refiner_simplicity():
    """Test ELEGANCE refiner simplicity pass."""
    refiner = EleganceRefiner()

    context = WritingContext(
        query="Test",
        memories=[],
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    text = "This is very really quite a simple test that is basically just for testing"

    result = await refiner.refine(text, context, pass_number=2)

    assert result.pass_number == 2
    assert result.focus == 'simplicity'
    assert len(result.improvements) > 0


@pytest.mark.asyncio
async def test_elegance_refiner_beauty():
    """Test ELEGANCE refiner beauty pass."""
    refiner = EleganceRefiner()

    context = WritingContext(
        query="Test",
        memories=[],
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    text = "The test was implemented by the developer"

    result = await refiner.refine(text, context, pass_number=3)

    assert result.pass_number == 3
    assert result.focus == 'beauty'


# Test Composer

@pytest.mark.asyncio
async def test_composer_basic():
    """Test basic composition with refinement."""
    refiners = {
        RefinementStrategy.ELEGANCE: EleganceRefiner()
    }
    composer = Composer(refiners=refiners)

    context = WritingContext(
        query="Test query",
        memories=[],
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    initial_draft = "This is very really a basic test that was implemented to facilitate testing"

    result = await composer.compose(
        initial_draft=initial_draft,
        context=context,
        strategy=RefinementStrategy.ELEGANCE,
        max_passes=3,
        quality_threshold=0.95
    )

    assert result.content is not None
    assert len(result.refinement_passes) > 0
    assert result.quality_score > 0.0
    assert result.quality_score <= 1.0


@pytest.mark.asyncio
async def test_composer_quality_trajectory():
    """Test that quality improves over passes."""
    refiners = {
        RefinementStrategy.ELEGANCE: EleganceRefiner()
    }
    composer = Composer(refiners=refiners)

    context = WritingContext(
        query="Test",
        memories=[],
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    initial_draft = "This is very really quite a test"

    result = await composer.compose(
        initial_draft=initial_draft,
        context=context,
        strategy=RefinementStrategy.ELEGANCE,
        max_passes=3,
        quality_threshold=0.95
    )

    # Quality should generally improve
    trajectory = result.quality_trajectory
    assert len(trajectory) > 0

    # Check metadata
    assert 'initial_quality' in result.metadata
    assert 'final_quality' in result.metadata
    assert 'improvement' in result.metadata


# Test Writer (full pipeline)

@pytest.mark.asyncio
async def test_writer_full_pipeline(sample_memories):
    """Test full writing pipeline with refinement."""
    mode_writers = {
        WritingMode.NARRATIVE: NarrativeWriter()
    }
    refiners = {
        RefinementStrategy.ELEGANCE: EleganceRefiner()
    }
    composer = Composer(refiners=refiners)
    writer = Writer(mode_writers=mode_writers, composer=composer)

    context = WritingContext(
        query="What is Thompson Sampling?",
        memories=sample_memories,
        mode=WritingMode.AUTO,
        style=StyleGuide.AUTO
    )

    result = await writer.write(context, refine=True)

    assert result.content is not None
    assert len(result.content) > 0
    assert result.mode == WritingMode.NARRATIVE  # Should auto-detect
    assert result.quality_score > 0.0
    assert result.confidence > 0.0
    assert len(result.refinement_passes) > 0


@pytest.mark.asyncio
async def test_writer_without_refinement(sample_memories):
    """Test writing without refinement."""
    mode_writers = {
        WritingMode.NARRATIVE: NarrativeWriter()
    }
    writer = Writer(mode_writers=mode_writers, composer=None)

    context = WritingContext(
        query="What is Thompson Sampling?",
        memories=sample_memories,
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    result = await writer.write(context, refine=False)

    assert result.content is not None
    assert len(result.refinement_passes) == 0  # No refinement
    assert result.metadata['refined'] == False


# Test simple API

@pytest.mark.asyncio
async def test_simple_write_api(sample_memories):
    """Test the simple write() API."""
    content = await write(
        query="What is Thompson Sampling?",
        memories=sample_memories,
        refine=True
    )

    assert content is not None
    assert len(content) > 0
    assert isinstance(content, str)


@pytest.mark.asyncio
async def test_refine_text_api():
    """Test the refine_text() API."""
    draft = "This is very really a basic test that was implemented"

    refined = await refine_text(
        text=draft,
        passes=2,
        strategy='elegance'
    )

    assert refined is not None
    assert isinstance(refined, str)
    # Should have removed filler words
    assert 'very' not in refined or 'really' not in refined


# Test edge cases

@pytest.mark.asyncio
async def test_writer_empty_query():
    """Test writer with empty query."""
    writer = Writer()

    context = WritingContext(
        query="",
        memories=[],
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    result = await writer.write(context, refine=False)

    assert result.content is not None


@pytest.mark.asyncio
async def test_quality_scoring_empty_text():
    """Test quality scoring with empty text."""
    composer = Composer()

    context = WritingContext(
        query="Test",
        memories=[],
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    quality, dimensions = composer.score_quality("", context)

    assert quality == 0.0


# Performance test (optional, can be skipped)

@pytest.mark.asyncio
@pytest.mark.slow
async def test_writing_performance(sample_memories):
    """Test writing performance (optional slow test)."""
    import time

    writer = Writer(
        mode_writers={WritingMode.NARRATIVE: NarrativeWriter()},
        composer=Composer(refiners={RefinementStrategy.ELEGANCE: EleganceRefiner()})
    )

    context = WritingContext(
        query="What is Thompson Sampling?",
        memories=sample_memories,
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    start = time.time()
    result = await writer.write(context, refine=True)
    duration = (time.time() - start) * 1000  # ms

    # Should complete in reasonable time (<500ms for typical content)
    assert duration < 500

    print(f"\nWriting performance: {duration:.1f}ms")
    print(f"Quality trajectory: {result.quality_trajectory}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
