"""
Optimize Strategy - Recursive Optimization

Implements Recursive Optimization technique where the prompt is
improved through multiple systematic iterations, each focusing on
a different improvement dimension.
"""

from pathlib import Path
import yaml
import logging

from HoloLoom.prompting.strategy import (
    PromptingStrategy,
    StrategyContext,
    StrategyResult,
    StrategyCategory
)

logger = logging.getLogger(__name__)


class OptimizeStrategy(PromptingStrategy):
    """
    Recursive Optimization strategy.

    Improves prompts through systematic iterations:
    1. Add missing constraints
    2. Resolve ambiguities
    3. Enhance reasoning depth

    Each iteration builds on the previous, tracking quality
    improvements along the way.

    Example:
        >>> strategy = OptimizeStrategy()
        >>> context = StrategyContext(query="write a function")
        >>> result = await strategy.enhance(context)
        >>> # Enhanced query includes 3-iteration optimization process
    """

    def __init__(self):
        """Initialize optimize strategy"""
        # Load config
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load template
        template_path = Path(__file__).parent / "template.md"
        with open(template_path, 'r') as f:
            self.template = f.read()

        logger.debug(
            f"Loaded optimize strategy "
            f"(iterations={self.config['config']['iterations']})"
        )

    @property
    def name(self) -> str:
        return "optimize"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.META_PROMPTING

    @property
    def description(self) -> str:
        return "Recursive Optimization: Multi-iteration prompt improvement"

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        """
        Apply recursive optimization template to query.

        Args:
            context: Query context

        Returns:
            Enhanced query with multi-iteration optimization process
        """
        # Format template
        enhanced = self.template.format(
            query=context.query,
            iterations=self.config['config']['iterations']
        )

        logger.info(
            f"Enhanced query with {self.config['config']['iterations']}-iteration "
            f"optimization process"
        )

        return StrategyResult(
            enhanced_query=enhanced,
            metadata={
                'strategy': 'optimize',
                'category': 'meta-prompting',
                'iterations': self.config['config']['iterations'],
                'improvement_focus': self.config['config']['improvement_focus'],
                'original_query': context.query
            },
            confidence=0.9,
            estimated_improvement=0.38
        )

    def can_apply(self, context: StrategyContext) -> float:
        """
        Calculate confidence that optimization is appropriate.

        Optimization is useful for:
        - Explicit optimization requests (very high confidence)
        - Improvement requests (high confidence)
        - Creation/generation tasks (medium confidence)
        - Complex queries (medium confidence)

        Args:
            context: Query context

        Returns:
            Confidence score 0-1
        """
        query_lower = context.query.lower()
        detection = self.config['detection']

        base_confidence = detection['base_confidence']
        keyword_boost = detection['keyword_boost']

        confidence = base_confidence

        # Check high confidence keywords (explicit optimization)
        high_keywords = detection['high_confidence_keywords']
        if any(word in query_lower for word in high_keywords):
            confidence += keyword_boost
            logger.debug(f"High-confidence keyword found in: {context.query[:50]}")

        # Check medium confidence keywords (creation tasks)
        medium_keywords = detection['medium_confidence_keywords']
        if any(word in query_lower for word in medium_keywords):
            confidence += keyword_boost * 0.5
            logger.debug(f"Medium-confidence keyword found in: {context.query[:50]}")

        # Slight penalty for very long queries (might not need optimization)
        if len(context.query) > 200:
            confidence -= detection['iteration_penalty']
            logger.debug("Applied iteration penalty for long query")

        # Cap at 1.0
        confidence = min(confidence, 1.0)

        logger.debug(
            f"Optimize strategy confidence for '{context.query[:50]}': "
            f"{confidence:.2f}"
        )

        return confidence
