"""
Basic Refiner
=============

Fallback refiner for strategies without specialized implementation.
"""

import logging
from typing import List

from ..core.protocol import (
    RefinerProtocol,
    RefinementPass,
    RefinementStrategy,
    WritingContext
)

logger = logging.getLogger(__name__)


class BasicRefiner:
    """
    Basic refiner that applies simple heuristic improvements.

    Used as fallback when specialized refiners aren't available.
    """

    def __init__(self, strategy: RefinementStrategy = RefinementStrategy.ELEGANCE):
        self.strategy = strategy
        self.logger = logger

    async def refine(
        self,
        text: str,
        context: WritingContext,
        pass_number: int
    ) -> RefinementPass:
        """
        Refine content in single pass using basic heuristics.

        Args:
            text: Content to refine
            context: Writing context
            pass_number: Current pass number

        Returns:
            Refinement pass result
        """
        improvements = []
        refined = text

        # Basic improvements that work for any strategy

        # 1. Fix double spaces
        if '  ' in refined:
            refined = ' '.join(refined.split())
            improvements.append("Fixed spacing")

        # 2. Ensure paragraph breaks exist
        if '\n\n' not in refined and len(refined) > 200:
            sentences = [s.strip() + '.' for s in refined.split('.') if s.strip()]
            if len(sentences) > 4:
                mid = len(sentences) // 2
                refined = '. '.join(sentences[:mid]) + '\n\n' + '. '.join(sentences[mid:])
                improvements.append("Added paragraph structure")

        # 3. Capitalize sentences
        refined_sentences = []
        for sentence in refined.split('. '):
            if sentence and sentence[0].islower():
                sentence = sentence[0].upper() + sentence[1:]
                improvements.append("Fixed capitalization")
            refined_sentences.append(sentence)
        refined = '. '.join(refined_sentences)

        # If no improvements, add a note
        if not improvements:
            improvements.append("Text already well-formed")

        focus = f"pass_{pass_number}"

        return RefinementPass(
            pass_number=pass_number,
            strategy=self.strategy_name,
            focus=focus,
            input_text=text,
            output_text=refined,
            quality_score=0.0,  # Will be filled by composer
            improvements=improvements,
            duration_ms=0.0  # Will be filled by composer
        )

    @property
    def strategy_name(self) -> str:
        """Strategy name for this refiner."""
        return f"basic_{self.strategy.value}"

    @property
    def passes(self) -> List[str]:
        """List of focus areas for each pass."""
        return ["pass_1", "pass_2", "pass_3"]
