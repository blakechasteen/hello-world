"""
Verify Strategy - Chain of Verification

Implements Chain of Verification (CoV) prompting technique.
Forces the model to critique its own output through multiple passes.
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


class VerifyStrategy(PromptingStrategy):
    """
    Chain of Verification strategy.

    Enhances queries by adding a 4-pass verification loop:
    1. Initial analysis
    2. Identify incompleteness
    3. Cite evidence
    4. Revised analysis

    This forces the model to find and fix gaps in its own reasoning,
    leading to higher quality outputs.

    Example:
        >>> strategy = VerifyStrategy()
        >>> context = StrategyContext(query="analyze this contract")
        >>> result = await strategy.enhance(context)
        >>> # Enhanced query now includes verification loop
    """

    def __init__(self):
        """Initialize verify strategy"""
        # Load config
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load template
        template_path = Path(__file__).parent / "template.md"
        with open(template_path, 'r') as f:
            self.template = f.read()

        logger.debug(f"Loaded verify strategy (passes={self.config['config']['passes']})")

    @property
    def name(self) -> str:
        return "verify"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.SELF_CORRECTION

    @property
    def description(self) -> str:
        return "Chain of Verification: Force model to critique its own output"

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        """
        Apply verification template to query.

        Args:
            context: Query context

        Returns:
            Enhanced query with verification loop
        """
        # Format template
        enhanced = self.template.format(
            query=context.query,
            passes=self.config['config']['passes'],
            depth=self.config['config']['depth']
        )

        logger.info(f"Enhanced query with {self.config['config']['passes']}-pass verification")

        return StrategyResult(
            enhanced_query=enhanced,
            metadata={
                'strategy': 'verify',
                'category': 'self-correction',
                'passes': self.config['config']['passes'],
                'depth': self.config['config']['depth'],
                'original_query': context.query
            },
            confidence=0.9,
            estimated_improvement=0.35
        )

    def can_apply(self, context: StrategyContext) -> float:
        """
        Calculate confidence that verification is appropriate.

        Verification is useful for:
        - Analysis tasks (high confidence)
        - Claims needing checking (high confidence)
        - Security/legal contexts (very high confidence)
        - General queries (medium confidence)

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

        # Check high confidence keywords
        high_keywords = detection['high_confidence_keywords']
        if any(word in query_lower for word in high_keywords):
            confidence += keyword_boost
            logger.debug(f"High-confidence keyword found in: {context.query[:50]}")

        # Check medium confidence keywords
        medium_keywords = detection['medium_confidence_keywords']
        if any(word in query_lower for word in medium_keywords):
            confidence += keyword_boost * 0.5
            logger.debug(f"Medium-confidence keyword found in: {context.query[:50]}")

        # Check file path context
        if context.file_path:
            file_patterns = detection['context_triggers']['file_patterns']
            for pattern in file_patterns:
                # Simple pattern matching (could be improved with pathlib.match)
                pattern_keyword = pattern.replace('*', '').replace('/', '')
                if pattern_keyword in context.file_path.lower():
                    confidence += context_boost
                    logger.debug(f"File path matches pattern: {pattern}")
                    break

        # Cap at 1.0
        confidence = min(confidence, 1.0)

        logger.debug(f"Verify strategy confidence for '{context.query[:50]}': {confidence:.2f}")

        return confidence
