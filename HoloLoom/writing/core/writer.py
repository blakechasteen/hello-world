"""
Core Writer Implementation
===========================

Main Writer class that orchestrates content generation from memory.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from .protocol import (
    WriterProtocol,
    WritingContext,
    WritingResult,
    WritingMode,
    StyleGuide,
    RefinementStrategy,
    RefinementPass,
    MemoryShard
)

logger = logging.getLogger(__name__)


class Writer:
    """
    Main writer class that transforms memory into prose.

    Philosophy: "Great writing isn't written, it's refined."

    The Writer:
    1. Detects appropriate mode and style (if AUTO)
    2. Generates initial draft using mode-specific writer
    3. Optionally refines through composer
    4. Returns final result with quality metrics
    """

    def __init__(
        self,
        mode_writers: Optional[Dict[WritingMode, Any]] = None,
        composer: Optional[Any] = None
    ):
        """
        Initialize writer.

        Args:
            mode_writers: Dict mapping WritingMode to ModeWriterProtocol instances
            composer: ComposerProtocol instance for refinement
        """
        self.mode_writers = mode_writers or {}
        self.composer = composer
        self.logger = logger

    async def write(
        self,
        context: WritingContext,
        refine: bool = True
    ) -> WritingResult:
        """
        Generate content from memory context.

        Args:
            context: Writing context with query and memories
            refine: Enable multi-pass refinement

        Returns:
            Writing result with content and metadata
        """
        start_time = time.time()

        # Auto-detect mode if needed
        if context.mode == WritingMode.AUTO:
            context.mode = self.detect_mode(context.query, context.memories)
            self.logger.info(f"Auto-detected mode: {context.mode.value}")

        # Auto-detect style if needed
        if context.style == StyleGuide.AUTO:
            context.style = self.detect_style(context.query, context.memories, context.mode)
            self.logger.info(f"Auto-detected style: {context.style.value}")

        # Generate initial draft
        initial_draft = await self._generate_draft(context)
        self.logger.debug(f"Generated initial draft: {len(initial_draft)} chars")

        # Refine if requested and composer available
        if refine and self.composer is not None:
            # Determine refinement strategy based on mode
            strategy = self._select_refinement_strategy(context.mode)

            result = await self.composer.compose(
                initial_draft=initial_draft,
                context=context,
                strategy=strategy,
                max_passes=3,
                quality_threshold=0.9
            )
        else:
            # No refinement - create result from initial draft
            quality, _ = self._score_quality(initial_draft, context)
            result = WritingResult(
                content=initial_draft,
                mode=context.mode,
                style=context.style,
                quality_score=quality,
                confidence=0.7,  # Lower confidence without refinement
                refinement_passes=[],
                metadata={
                    'refined': False,
                    'duration_ms': (time.time() - start_time) * 1000
                }
            )

        # Add timing metadata
        result.metadata['total_duration_ms'] = (time.time() - start_time) * 1000

        self.logger.info(
            f"Generated {context.mode.value} content: "
            f"{len(result.content)} chars, quality={result.quality_score:.2f}"
        )

        return result

    async def _generate_draft(self, context: WritingContext) -> str:
        """
        Generate initial draft using mode-specific writer.

        Args:
            context: Writing context

        Returns:
            Initial draft content
        """
        # Get mode-specific writer
        mode_writer = self.mode_writers.get(context.mode)

        if mode_writer is not None:
            # Use specialized mode writer
            return await mode_writer.generate(context)
        else:
            # Fallback to basic template-based generation
            return self._generate_fallback(context)

    def _generate_fallback(self, context: WritingContext) -> str:
        """
        Fallback generation when no mode-specific writer available.

        Args:
            context: Writing context

        Returns:
            Basic generated content
        """
        # Extract key information from memories
        if not context.memories:
            return f"No relevant information found for: {context.query}"

        # Simple concatenation with structure
        parts = [f"# {context.query}\n"]

        # Group memories by relevance (if metadata available)
        sorted_memories = sorted(
            context.memories,
            key=lambda m: m.metadata.get('relevance', 0.5),
            reverse=True
        )

        # Take top memories and format
        for i, memory in enumerate(sorted_memories[:5]):
            content = memory.text.strip()
            if content:
                parts.append(f"\n## Point {i+1}\n\n{content}")

        return "\n".join(parts)

    def detect_mode(
        self,
        query: str,
        memories: List[MemoryShard]
    ) -> WritingMode:
        """
        Auto-detect appropriate writing mode.

        Args:
            query: User query
            memories: Memory context

        Returns:
            Detected writing mode
        """
        query_lower = query.lower()

        # Technical indicators
        technical_keywords = [
            'how to', 'implementation', 'code', 'api', 'function',
            'documentation', 'technical', 'architecture', 'system'
        ]
        if any(kw in query_lower for kw in technical_keywords):
            return WritingMode.TECHNICAL

        # Analysis indicators
        analysis_keywords = [
            'analyze', 'compare', 'evaluate', 'assess', 'report',
            'summary', 'overview', 'breakdown', 'study'
        ]
        if any(kw in query_lower for kw in analysis_keywords):
            return WritingMode.ANALYSIS

        # Creative indicators
        creative_keywords = [
            'story', 'poem', 'creative', 'fiction', 'narrative',
            'imagine', 'create', 'write a story'
        ]
        if any(kw in query_lower for kw in creative_keywords):
            return WritingMode.CREATIVE

        # Dialogue indicators
        dialogue_keywords = [
            'conversation', 'chat', 'discuss', 'dialogue', 'talk about'
        ]
        if any(kw in query_lower for kw in dialogue_keywords):
            return WritingMode.DIALOGUE

        # Check for question words -> narrative explanation
        question_words = ['what', 'why', 'how', 'when', 'where', 'who']
        if any(query_lower.startswith(qw) for qw in question_words):
            return WritingMode.NARRATIVE

        # Default to narrative
        return WritingMode.NARRATIVE

    def detect_style(
        self,
        query: str,
        memories: List[MemoryShard],
        mode: WritingMode
    ) -> StyleGuide:
        """
        Auto-detect appropriate style guide.

        Args:
            query: User query
            memories: Memory context
            mode: Writing mode

        Returns:
            Detected style guide
        """
        # Mode-specific defaults
        mode_style_map = {
            WritingMode.TECHNICAL: StyleGuide.TECHNICAL,
            WritingMode.CREATIVE: StyleGuide.CREATIVE,
            WritingMode.ANALYSIS: StyleGuide.ACADEMIC,
            WritingMode.DIALOGUE: StyleGuide.CASUAL,
            WritingMode.NARRATIVE: StyleGuide.TUFTE,  # Clarity-first
            WritingMode.CODE_DOC: StyleGuide.TECHNICAL
        }

        # Check query for style hints
        query_lower = query.lower()

        if 'academic' in query_lower or 'research' in query_lower:
            return StyleGuide.ACADEMIC

        if 'casual' in query_lower or 'simple' in query_lower or 'eli5' in query_lower:
            return StyleGuide.CASUAL

        if 'clear' in query_lower or 'concise' in query_lower:
            return StyleGuide.TUFTE

        # Use mode default
        return mode_style_map.get(mode, StyleGuide.TUFTE)

    def _select_refinement_strategy(self, mode: WritingMode) -> RefinementStrategy:
        """
        Select appropriate refinement strategy for mode.

        Args:
            mode: Writing mode

        Returns:
            Refinement strategy
        """
        # Technical/analysis needs verification
        if mode in [WritingMode.TECHNICAL, WritingMode.ANALYSIS, WritingMode.CODE_DOC]:
            return RefinementStrategy.VERIFY

        # Creative/narrative benefits from elegance
        if mode in [WritingMode.CREATIVE, WritingMode.NARRATIVE]:
            return RefinementStrategy.ELEGANCE

        # Dialogue needs coherence
        if mode == WritingMode.DIALOGUE:
            return RefinementStrategy.COHERENCE

        # Default to elegance
        return RefinementStrategy.ELEGANCE

    def _score_quality(
        self,
        text: str,
        context: WritingContext
    ) -> tuple[float, Dict[str, float]]:
        """
        Score content quality (simple heuristic version).

        Args:
            text: Content to score
            context: Writing context

        Returns:
            (overall_quality, dimension_scores)
        """
        if not text:
            return 0.0, {}

        scores = {}

        # Clarity: sentence length variance (lower is clearer)
        sentences = text.split('.')
        if sentences:
            avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
            scores['clarity'] = max(0.0, min(1.0, 1.0 - (avg_sentence_len - 15) / 30))
        else:
            scores['clarity'] = 0.5

        # Completeness: content length (relative to query complexity)
        min_length = max(1, len(context.query.split()) * 10)  # Expect 10x expansion, min 1
        scores['completeness'] = min(1.0, len(text.split()) / min_length)

        # Coherence: paragraph structure
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            scores['coherence'] = min(1.0, len(paragraphs) / 5)  # Good if 3-5 paragraphs
        else:
            scores['coherence'] = 0.5

        # Overall: weighted average
        overall = (
            0.4 * scores.get('clarity', 0.5) +
            0.3 * scores.get('completeness', 0.5) +
            0.3 * scores.get('coherence', 0.5)
        )

        return overall, scores


# Standalone write function for simple API
async def write(
    query: str,
    memories: List[MemoryShard],
    mode: Optional[WritingMode] = None,
    style: Optional[StyleGuide] = None,
    refine: bool = True
) -> str:
    """
    Simple API for content generation.

    Args:
        query: What to write about
        memories: Memory context
        mode: Writing mode (auto-detected if None)
        style: Style guide (auto-detected if None)
        refine: Enable refinement

    Returns:
        Generated content
    """
    context = WritingContext(
        query=query,
        memories=memories,
        mode=mode or WritingMode.AUTO,
        style=style or StyleGuide.AUTO
    )

    writer = Writer()
    result = await writer.write(context, refine=refine)

    return result.content
