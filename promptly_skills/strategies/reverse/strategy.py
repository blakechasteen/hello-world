"""
Reverse Strategy - Reverse Prompting

Implements Reverse Prompting technique where we ask the model to
design its own optimal prompt rather than answering directly.
This leverages the model's meta-knowledge about prompting.
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


class ReverseStrategy(PromptingStrategy):
    """
    Reverse Prompting strategy.

    Instead of answering the user's query directly, we ask the model
    to design the optimal prompt for that query. This exploits the
    model's meta-knowledge about what makes prompts effective.

    The model analyzes the request, designs a comprehensive prompt
    using the 7-component framework, justifies its design choices,
    and optionally executes the prompt.

    Example:
        >>> strategy = ReverseStrategy()
        >>> context = StrategyContext(query="help me understand SQL")
        >>> result = await strategy.enhance(context)
        >>> # Enhanced query asks model to design optimal SQL teaching prompt
    """

    def __init__(self):
        """Initialize reverse strategy"""
        # Load config
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load template
        template_path = Path(__file__).parent / "template.md"
        with open(template_path, 'r') as f:
            self.template = f.read()

        logger.debug(
            f"Loaded reverse strategy "
            f"(design_approach={self.config['config']['design_approach']})"
        )

    @property
    def name(self) -> str:
        return "reverse"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.META_PROMPTING

    @property
    def description(self) -> str:
        return "Reverse Prompting: Model designs its own optimal prompt"

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        """
        Apply reverse prompting template to query.

        Args:
            context: Query context

        Returns:
            Enhanced query asking model to design optimal prompt
        """
        # Format template
        enhanced = self.template.format(
            query=context.query
        )

        logger.info("Enhanced query with reverse prompt engineering")

        return StrategyResult(
            enhanced_query=enhanced,
            metadata={
                'strategy': 'reverse',
                'category': 'meta-prompting',
                'design_approach': self.config['config']['design_approach'],
                'include_justification': self.config['config']['include_justification'],
                'original_query': context.query
            },
            confidence=0.95,
            estimated_improvement=0.45
        )

    def can_apply(self, context: StrategyContext) -> float:
        """
        Calculate confidence that reverse prompting is appropriate.

        Reverse prompting is useful for:
        - Explicit prompt design requests (very high confidence)
        - Meta-level "how should I ask" questions (high confidence)
        - Vague/unclear requests (medium-high confidence)
        - "Help me" questions (medium confidence)

        Args:
            context: Query context

        Returns:
            Confidence score 0-1
        """
        query_lower = context.query.lower()
        detection = self.config['detection']

        base_confidence = detection['base_confidence']
        keyword_boost = detection['keyword_boost']
        meta_boost = detection['meta_boost']

        confidence = base_confidence

        # Check high confidence keywords (explicit prompt design)
        high_keywords = detection['high_confidence_keywords']
        if any(word in query_lower for word in high_keywords):
            confidence += keyword_boost
            logger.debug(f"High-confidence keyword found in: {context.query[:50]}")

        # Check medium confidence keywords (meta-level questions)
        medium_keywords = detection['medium_confidence_keywords']
        if any(word in query_lower for word in medium_keywords):
            confidence += keyword_boost * 0.5
            logger.debug(f"Medium-confidence keyword found in: {context.query[:50]}")

        # Boost for meta-prompting keywords (prompt, ask, say)
        meta_keywords = ['prompt', 'prompting', 'ask', 'query']
        if any(word in query_lower for word in meta_keywords):
            confidence += meta_boost
            logger.debug("Meta-prompting keyword found")

        # Cap at 1.0
        confidence = min(confidence, 1.0)

        logger.debug(
            f"Reverse strategy confidence for '{context.query[:50]}': "
            f"{confidence:.2f}"
        )

        return confidence
