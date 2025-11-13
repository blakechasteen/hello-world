"""
Challenge Strategy - Adversarial Prompting

Implements Adversarial Prompting technique where we demand the model
find problems even if it needs to stretch. Forces aggressive security
analysis and worst-case thinking.
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


class ChallengeStrategy(PromptingStrategy):
    """
    Adversarial Challenge strategy.

    Forces the model to attack the provided work and find problems.
    Must identify minimum number of specific vulnerabilities with
    exploitation scenarios and mitigations.

    This is more aggressive than verification - we demand finding
    problems even if the model needs to stretch or be paranoid.

    Example:
        >>> strategy = ChallengeStrategy()
        >>> context = StrategyContext(query="review security architecture")
        >>> result = await strategy.enhance(context)
        >>> # Enhanced query demands finding 5+ vulnerabilities
    """

    def __init__(self):
        """Initialize challenge strategy"""
        # Load config
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load template
        template_path = Path(__file__).parent / "template.md"
        with open(template_path, 'r') as f:
            self.template = f.read()

        logger.debug(
            f"Loaded challenge strategy "
            f"(min_problems={self.config['config']['min_problems']})"
        )

    @property
    def name(self) -> str:
        return "challenge"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.SELF_CORRECTION

    @property
    def description(self) -> str:
        return "Adversarial Prompting: Demand finding problems even if stretching"

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        """
        Apply adversarial challenge template to query.

        Args:
            context: Query context

        Returns:
            Enhanced query with adversarial challenge prompt
        """
        # Format template
        enhanced = self.template.format(
            query=context.query,
            min_problems=self.config['config']['min_problems'],
            attack_style=self.config['config']['attack_style']
        )

        logger.info(
            f"Enhanced query with adversarial challenge "
            f"(must find {self.config['config']['min_problems']}+ problems)"
        )

        return StrategyResult(
            enhanced_query=enhanced,
            metadata={
                'strategy': 'challenge',
                'category': 'self-correction',
                'min_problems': self.config['config']['min_problems'],
                'attack_style': self.config['config']['attack_style'],
                'original_query': context.query
            },
            confidence=0.95,
            estimated_improvement=0.62
        )

    def can_apply(self, context: StrategyContext) -> float:
        """
        Calculate confidence that adversarial challenge is appropriate.

        Challenge is most useful for:
        - Security reviews (very high confidence)
        - Authentication systems (very high confidence)
        - APIs and interfaces (high confidence)
        - Any audit/test task (medium-high confidence)

        Args:
            context: Query context

        Returns:
            Confidence score 0-1
        """
        query_lower = context.query.lower()
        detection = self.config['detection']

        base_confidence = detection['base_confidence']
        keyword_boost = detection['keyword_boost']
        context_boost = detection['context_boost']

        confidence = base_confidence

        # Check high confidence keywords (security-focused)
        high_keywords = detection['high_confidence_keywords']
        if any(word in query_lower for word in high_keywords):
            confidence += keyword_boost
            logger.debug(f"High-confidence keyword found in: {context.query[:50]}")

        # Check medium confidence keywords (general review)
        medium_keywords = detection['medium_confidence_keywords']
        if any(word in query_lower for word in medium_keywords):
            confidence += keyword_boost * 0.5
            logger.debug(f"Medium-confidence keyword found in: {context.query[:50]}")

        # Check file path context
        if context.file_path:
            file_patterns = detection['context_triggers']['file_patterns']
            for pattern in file_patterns:
                pattern_keyword = pattern.replace('*', '').replace('/', '')
                if pattern_keyword in context.file_path.lower():
                    confidence += context_boost
                    logger.debug(f"File path matches pattern: {pattern}")
                    break

        # Cap at 1.0
        confidence = min(confidence, 1.0)

        logger.debug(
            f"Challenge strategy confidence for '{context.query[:50]}': "
            f"{confidence:.2f}"
        )

        return confidence
