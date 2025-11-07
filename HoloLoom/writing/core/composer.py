"""
Multi-Pass Composer
===================

Composer that orchestrates refinement across multiple passes.

Philosophy: "Great writing isn't written, it's refined."
"""

import logging
import time
from typing import List, Dict, Any, Optional

from .protocol import (
    ComposerProtocol,
    WritingContext,
    WritingResult,
    RefinementStrategy,
    RefinementPass,
    QUALITY_DIMENSIONS
)

logger = logging.getLogger(__name__)


class Composer:
    """
    Multi-pass composer that refines content iteratively.

    Inspired by HoloLoom's recursive learning:
    1. Generate initial draft (low quality)
    2. Multi-pass refinement (ELEGANCE/VERIFY strategies)
    3. Quality scoring (confidence → coherence → completeness)
    4. Learning (learns which refinements work)
    """

    def __init__(self, refiners: Optional[Dict[RefinementStrategy, Any]] = None):
        """
        Initialize composer.

        Args:
            refiners: Dict mapping RefinementStrategy to RefinerProtocol instances
        """
        self.refiners = refiners or {}
        self.logger = logger

    async def compose(
        self,
        initial_draft: str,
        context: WritingContext,
        strategy: RefinementStrategy = RefinementStrategy.ELEGANCE,
        max_passes: int = 3,
        quality_threshold: float = 0.9
    ) -> WritingResult:
        """
        Multi-pass composition with refinement.

        Args:
            initial_draft: Initial content draft
            context: Writing context
            strategy: Refinement strategy
            max_passes: Maximum refinement passes
            quality_threshold: Target quality score

        Returns:
            Writing result with refinement history
        """
        start_time = time.time()
        refinement_passes: List[RefinementPass] = []
        current_text = initial_draft

        # Get refiner for strategy
        refiner = self.refiners.get(strategy)

        if refiner is None:
            # No refiner available - use built-in basic refinement
            self.logger.warning(f"No refiner for {strategy.value}, using basic refinement")
            refiner = self._get_basic_refiner(strategy)

        # Initial quality score
        initial_quality, initial_dimensions = self.score_quality(initial_draft, context)
        self.logger.info(
            f"Initial quality: {initial_quality:.2f} "
            f"(clarity={initial_dimensions.get('clarity', 0):.2f}, "
            f"completeness={initial_dimensions.get('completeness', 0):.2f})"
        )

        # Refinement loop
        for pass_num in range(1, max_passes + 1):
            pass_start = time.time()

            # Refine
            refinement_pass = await refiner.refine(
                text=current_text,
                context=context,
                pass_number=pass_num
            )

            # Update duration
            refinement_pass.duration_ms = (time.time() - pass_start) * 1000

            # Score quality after this pass
            quality, dimensions = self.score_quality(refinement_pass.output_text, context)
            refinement_pass.quality_score = quality

            refinement_passes.append(refinement_pass)
            current_text = refinement_pass.output_text

            self.logger.info(
                f"Pass {pass_num} ({refinement_pass.focus}): "
                f"quality {quality:.2f}, "
                f"{len(refinement_pass.improvements)} improvements"
            )

            # Check if we've reached threshold
            if quality >= quality_threshold:
                self.logger.info(f"Reached quality threshold {quality_threshold:.2f}")
                break

            # Check for diminishing returns
            if pass_num > 1:
                prev_quality = refinement_passes[-2].quality_score
                improvement = quality - prev_quality
                if improvement < 0.02:  # Less than 2% improvement
                    self.logger.info("Diminishing returns detected, stopping refinement")
                    break

        # Calculate final metrics
        final_quality = refinement_passes[-1].quality_score if refinement_passes else initial_quality
        confidence = self._calculate_confidence(refinement_passes)

        result = WritingResult(
            content=current_text,
            mode=context.mode,
            style=context.style,
            quality_score=final_quality,
            confidence=confidence,
            refinement_passes=refinement_passes,
            metadata={
                'strategy': strategy.value,
                'initial_quality': initial_quality,
                'final_quality': final_quality,
                'improvement': final_quality - initial_quality,
                'passes_taken': len(refinement_passes),
                'duration_ms': (time.time() - start_time) * 1000
            }
        )

        return result

    def score_quality(
        self,
        text: str,
        context: WritingContext
    ) -> tuple[float, Dict[str, float]]:
        """
        Score content quality across multiple dimensions.

        Args:
            text: Content to score
            context: Writing context

        Returns:
            (overall_quality, dimension_scores)
        """
        if not text:
            return 0.0, {}

        scores = {}

        # Clarity: sentence length and structure
        scores['clarity'] = self._score_clarity(text)

        # Completeness: content depth relative to query
        scores['completeness'] = self._score_completeness(text, context)

        # Coherence: paragraph structure and flow
        scores['coherence'] = self._score_coherence(text)

        # Accuracy: information consistency (basic check)
        scores['accuracy'] = self._score_accuracy(text, context)

        # Conciseness: signal-to-noise ratio
        scores['conciseness'] = self._score_conciseness(text)

        # Elegance: stylistic quality
        scores['elegance'] = self._score_elegance(text)

        # Overall: weighted average based on context mode
        weights = self._get_dimension_weights(context)
        overall = sum(scores.get(dim, 0.5) * weight for dim, weight in weights.items())

        return overall, scores

    def _score_clarity(self, text: str) -> float:
        """Score clarity based on sentence structure."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.5

        # Ideal sentence length: 15-25 words
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        clarity = 1.0 - min(1.0, abs(avg_length - 20) / 20)

        return max(0.3, clarity)  # Floor at 0.3

    def _score_completeness(self, text: str, context: WritingContext) -> float:
        """Score completeness based on content depth."""
        # Expect roughly 10x expansion from query
        query_words = len(context.query.split())
        text_words = len(text.split())
        expected_min = query_words * 10

        completeness = min(1.0, text_words / expected_min)
        return max(0.3, completeness)

    def _score_coherence(self, text: str) -> float:
        """Score coherence based on structure."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if len(paragraphs) < 2:
            return 0.5  # Single paragraph is less coherent

        # Ideal: 3-5 paragraphs with balanced length
        paragraph_score = min(1.0, len(paragraphs) / 5)

        # Check paragraph length variance (lower is better)
        if paragraphs:
            lengths = [len(p.split()) for p in paragraphs]
            avg_len = sum(lengths) / len(lengths)
            variance = sum(abs(l - avg_len) for l in lengths) / len(lengths)
            variance_score = 1.0 - min(1.0, variance / avg_len)
        else:
            variance_score = 0.5

        coherence = 0.6 * paragraph_score + 0.4 * variance_score
        return max(0.3, coherence)

    def _score_accuracy(self, text: str, context: WritingContext) -> float:
        """Score accuracy based on memory alignment."""
        # Simple heuristic: check if key terms from memories appear in text
        if not context.memories:
            return 0.7  # Neutral if no memories to check

        # Extract key terms from memories
        memory_terms = set()
        for memory in context.memories:
            words = memory.text.lower().split()
            # Take significant words (>4 chars)
            memory_terms.update(w for w in words if len(w) > 4)

        if not memory_terms:
            return 0.7

        # Count how many appear in text
        text_lower = text.lower()
        present = sum(1 for term in memory_terms if term in text_lower)
        accuracy = min(1.0, present / min(len(memory_terms), 10))  # Cap at 10 terms

        return max(0.5, accuracy)

    def _score_conciseness(self, text: str) -> float:
        """Score conciseness based on signal-to-noise."""
        # Penalize excessive filler words
        filler_words = {
            'very', 'really', 'quite', 'just', 'actually', 'basically',
            'literally', 'essentially', 'practically', 'virtually'
        }

        words = text.lower().split()
        if not words:
            return 0.5

        filler_count = sum(1 for w in words if w in filler_words)
        filler_ratio = filler_count / len(words)

        # Low filler ratio = high conciseness
        conciseness = 1.0 - min(1.0, filler_ratio * 10)  # Scale up penalty
        return max(0.3, conciseness)

    def _score_elegance(self, text: str) -> float:
        """Score elegance based on stylistic markers."""
        # Simple heuristics for elegance:
        # - Varied sentence structure
        # - Active voice preference
        # - Strong verbs

        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.5

        # Sentence length variance (good)
        lengths = [len(s.split()) for s in sentences]
        if len(lengths) > 1:
            avg_len = sum(lengths) / len(lengths)
            variance = sum(abs(l - avg_len) for l in lengths) / len(lengths)
            variety_score = min(1.0, variance / 5)  # Higher variance = better
        else:
            variety_score = 0.5

        # Check for passive voice markers (penalty)
        passive_markers = ['was', 'were', 'been', 'being']
        passive_count = sum(1 for word in text.lower().split() if word in passive_markers)
        passive_ratio = passive_count / len(text.split())
        active_score = 1.0 - min(1.0, passive_ratio * 5)

        elegance = 0.5 * variety_score + 0.5 * active_score
        return max(0.3, elegance)

    def _get_dimension_weights(self, context: WritingContext) -> Dict[str, float]:
        """Get quality dimension weights based on context."""
        # Default weights
        weights = {
            'clarity': 0.25,
            'completeness': 0.20,
            'coherence': 0.20,
            'accuracy': 0.15,
            'conciseness': 0.10,
            'elegance': 0.10
        }

        # Adjust based on mode
        from .protocol import WritingMode

        if context.mode == WritingMode.TECHNICAL:
            # Emphasize accuracy and completeness
            weights['accuracy'] = 0.30
            weights['completeness'] = 0.25
            weights['elegance'] = 0.05

        elif context.mode == WritingMode.CREATIVE:
            # Emphasize elegance and coherence
            weights['elegance'] = 0.30
            weights['coherence'] = 0.25
            weights['accuracy'] = 0.05

        elif context.mode == WritingMode.ANALYSIS:
            # Emphasize accuracy and clarity
            weights['accuracy'] = 0.30
            weights['clarity'] = 0.30

        return weights

    def _calculate_confidence(self, refinement_passes: List[RefinementPass]) -> float:
        """Calculate confidence based on refinement trajectory."""
        if not refinement_passes:
            return 0.5

        # High confidence if:
        # 1. Final quality is high
        # 2. Quality improved consistently
        # 3. Multiple passes completed

        final_quality = refinement_passes[-1].quality_score

        # Check improvement trajectory
        if len(refinement_passes) > 1:
            improvements = [
                refinement_passes[i].quality_score - refinement_passes[i-1].quality_score
                for i in range(1, len(refinement_passes))
            ]
            consistent = all(imp >= 0 for imp in improvements)
            trajectory_bonus = 0.1 if consistent else 0.0
        else:
            trajectory_bonus = 0.0

        # Passes bonus (more passes = more refinement)
        passes_bonus = min(0.1, len(refinement_passes) * 0.03)

        confidence = min(1.0, final_quality + trajectory_bonus + passes_bonus)
        return confidence

    def _get_basic_refiner(self, strategy: RefinementStrategy):
        """Get basic built-in refiner for strategy."""
        from ..refinement.basic import BasicRefiner
        return BasicRefiner(strategy=strategy)
