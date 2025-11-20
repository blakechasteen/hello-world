"""
HoloLoom Writing System
=======================
Universal content generation - memory becomes prose.

Philosophy: "Great writing isn't written, it's refined."

Ruthlessly Elegant API:
    from HoloLoom.writing import write

    # Generate content from memory
    content = await write(query, memories)

    # Automatic mode detection, refinement, and formatting

Advanced API:
    from HoloLoom.writing import Writer, Composer
    from HoloLoom.writing.core import WritingContext, WritingMode
    from HoloLoom.writing.refinement import EleganceRefiner
    from HoloLoom.writing.modes import NarrativeWriter

    # Custom configuration
    writer = Writer(
        mode_writers={WritingMode.NARRATIVE: NarrativeWriter()},
        composer=Composer(refiners={...})
    )

    result = await writer.write(context, refine=True)
"""

from typing import List, Optional

# Import primary API
from .core.writer import write as _write
from .core.protocol import MemoryShard

# Import core classes for advanced usage
from .core import (
    Writer,
    Composer,
    WritingContext,
    WritingResult,
    WritingMode,
    RefinementStrategy,
    StyleGuide,
    OutputFormat
)

# Import mode writers
from .modes import NarrativeWriter, TechnicalWriter, AnalysisWriter, CreativeWriter

# Import refiners
from .refinement import EleganceRefiner, BasicRefiner, VerifyRefiner


# Ruthlessly simple write function
async def write(
    query: str,
    memories: List[MemoryShard],
    mode: Optional[str] = None,
    style: Optional[str] = None,
    refine: bool = True,
    format: str = 'markdown'
) -> str:
    """
    Generate content from memory context.

    This is the primary API - ruthlessly simple.

    Args:
        query: What to write about
        memories: Memory context
        mode: Writing mode (narrative, technical, creative, etc.) - auto-detected if None
        style: Style guide (tufte, academic, casual) - auto-detected if None
        refine: Enable multi-pass refinement (default: True)
        format: Output format (default: 'markdown')

    Returns:
        Generated content

    Example:
        >>> from HoloLoom.writing import write
        >>> from HoloLoom.Documentation.types import MemoryShard
        >>>
        >>> memories = [
        ...     MemoryShard(
        ...         content="Thompson Sampling is a Bayesian approach to exploration",
        ...         metadata={'relevance': 0.9}
        ...     )
        ... ]
        >>>
        >>> content = await write(
        ...     "What is Thompson Sampling?",
        ...     memories,
        ...     refine=True
        ... )
    """
    # Convert string enums to actual enums
    mode_enum = WritingMode[mode.upper()] if mode else WritingMode.AUTO
    style_enum = StyleGuide[style.upper()] if style else StyleGuide.AUTO

    # Create context
    context = WritingContext(
        query=query,
        memories=memories,
        mode=mode_enum,
        style=style_enum
    )

    # Create writer with default configuration
    # (mode writers and refiners will be lazy-loaded)
    writer = create_default_writer()

    # Generate
    result = await writer.write(context, refine=refine)

    # Format output (for now, just return content)
    # Future: implement format converters
    return result.content


async def write_batch(
    queries: List[str],
    memories: List[MemoryShard],
    mode: Optional[str] = None,
    style: Optional[str] = None,
    refine: bool = True
) -> List[str]:
    """
    Batch writing with shared memory context.

    Args:
        queries: List of queries
        memories: Shared memory context
        mode: Writing mode
        style: Style guide
        refine: Enable refinement

    Returns:
        List of generated content
    """
    results = []
    for query in queries:
        content = await write(query, memories, mode, style, refine)
        results.append(content)
    return results


async def refine_text(
    text: str,
    query: str = "Refine this text",
    passes: int = 3,
    strategy: str = 'elegance'
) -> str:
    """
    Refine existing text.

    Args:
        text: Text to refine
        query: Context query (optional)
        passes: Number of refinement passes
        strategy: Refinement strategy (elegance, verify, etc.)

    Returns:
        Refined text
    """
    # Create dummy memory from input text
    memory = MemoryShard(id='refine_input', text=text, metadata={'relevance': 1.0})

    # Create context
    context = WritingContext(
        query=query,
        memories=[memory],
        mode=WritingMode.AUTO,
        style=StyleGuide.AUTO
    )

    # Create composer
    strategy_enum = RefinementStrategy[strategy.upper()]
    refiners = {strategy_enum: _get_refiner_for_strategy(strategy_enum)}
    composer = Composer(refiners=refiners)

    # Refine
    result = await composer.compose(
        initial_draft=text,
        context=context,
        strategy=strategy_enum,
        max_passes=passes,
        quality_threshold=0.95
    )

    return result.content


def create_default_writer() -> Writer:
    """
    Create writer with default configuration.

    Returns:
        Configured Writer instance
    """
    # Create mode writers (Phase 2: All modes implemented)
    mode_writers = {
        WritingMode.NARRATIVE: NarrativeWriter(),
        WritingMode.TECHNICAL: TechnicalWriter(),
        WritingMode.ANALYSIS: AnalysisWriter(),
        WritingMode.CREATIVE: CreativeWriter(),
    }

    # Create refiners (Phase 2: ELEGANCE + VERIFY)
    refiners = {
        RefinementStrategy.ELEGANCE: EleganceRefiner(),
        RefinementStrategy.VERIFY: VerifyRefiner(),
    }

    # Create composer
    composer = Composer(refiners=refiners)

    # Create writer
    return Writer(mode_writers=mode_writers, composer=composer)


def _get_refiner_for_strategy(strategy: RefinementStrategy):
    """Get refiner instance for strategy."""
    if strategy == RefinementStrategy.ELEGANCE:
        return EleganceRefiner()
    elif strategy == RefinementStrategy.VERIFY:
        return VerifyRefiner()
    else:
        return BasicRefiner(strategy=strategy)


__all__ = [
    # Primary API (ruthlessly simple)
    'write',
    'write_batch',
    'refine_text',

    # Advanced API
    'Writer',
    'Composer',
    'WritingContext',
    'WritingResult',

    # Enums
    'WritingMode',
    'RefinementStrategy',
    'StyleGuide',
    'OutputFormat',

    # Mode writers
    'NarrativeWriter',

    # Refiners
    'EleganceRefiner',
    'BasicRefiner',

    # Factory
    'create_default_writer'
]
