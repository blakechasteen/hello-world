"""
Deep Strategy - Deliberate Over-Instruction

Forces exhaustive depth on all aspects of a topic.
Covers: fundamentals, edge cases, tradeoffs, alternatives, examples, pitfalls, best practices.
"""

from pathlib import Path
from typing import Dict, Any
import yaml

from HoloLoom.prompting.strategy import (
    TemplateStrategy,
    StrategyContext,
    StrategyResult,
    StrategyCategory
)


class DeepStrategy(TemplateStrategy):
    """
    Deliberate Over-Instruction strategy.

    Forces model to go exhaustively deep on every aspect:
    - Fundamentals from first principles
    - Edge cases and boundary conditions
    - Tradeoffs (pros/cons/when to use)
    - Alternatives and comparisons
    - Concrete examples
    - Common pitfalls
    - Best practices

    Auto-detects:
    - High confidence (0.8) for depth keywords ("exhaustive", "in depth")
    - Medium confidence (0.6) for explanation requests ("explain", "understand")
    - Penalty for brevity keywords ("quick", "summary")

    Example:
        >>> strategy = DeepStrategy()
        >>> context = StrategyContext(query="explain neural networks")
        >>> result = await strategy.enhance(context)
        >>> # Result includes 7-section exhaustive analysis
    """

    def __init__(self):
        """Initialize deep strategy with config and template."""
        # Load config
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self._full_config = yaml.safe_load(f)

        # Load template
        template_path = Path(__file__).parent / "template.md"
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        # Initialize parent with template
        super().__init__(template, self._full_config['config'])

    @property
    def name(self) -> str:
        """Strategy name."""
        return "deep"

    @property
    def category(self) -> StrategyCategory:
        """Strategy category."""
        return StrategyCategory.META_PROMPTING

    @property
    def description(self) -> str:
        """Strategy description."""
        return "Deliberate Over-Instruction - Force exhaustive depth on every aspect"

    def can_apply(self, context: StrategyContext) -> float:
        """
        Determine if this strategy is appropriate for the context.

        High confidence (0.8+) for explicit depth requests.
        Medium confidence (0.6+) for explanation requests.
        Low confidence with brevity penalty for quick/summary requests.

        Args:
            context: Strategy context with query

        Returns:
            Confidence score (0-1)
        """
        query_lower = context.query.lower()
        detection_config = self._full_config['detection']

        # Start with base confidence
        confidence = detection_config['base_confidence']

        # Check for high confidence keywords
        high_keywords = detection_config['high_confidence_keywords']
        if any(keyword in query_lower for keyword in high_keywords):
            confidence += detection_config['keyword_boost']
            confidence += detection_config['depth_indicator_boost']

        # Check for medium confidence keywords
        elif any(keyword in query_lower for keyword in detection_config['medium_confidence_keywords']):
            confidence += detection_config['keyword_boost']

        # Check for brevity indicators (penalty)
        low_indicators = detection_config['low_confidence_indicators']
        if any(indicator in query_lower for indicator in low_indicators):
            confidence += detection_config['brevity_penalty']

        # Additional boost for "how" questions (often need depth)
        if query_lower.startswith('how ') or ' how ' in query_lower:
            confidence += 0.1

        # Additional boost for "explain" (common depth request)
        if 'explain' in query_lower:
            confidence += 0.15

        # Ensure in valid range
        return max(0.0, min(1.0, confidence))

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        """
        Enhance query with exhaustive depth instructions.

        Args:
            context: Strategy context with query

        Returns:
            Strategy result with enhanced query
        """
        # Format template with query
        enhanced_query = self.template.format(query=context.query)

        # Return result with metadata
        return StrategyResult(
            enhanced_query=enhanced_query,
            metadata={
                'strategy': 'deep',
                'sections': 7,
                'min_sections': self._full_config['config']['min_sections'],
                'min_depth_per_section': self._full_config['config']['min_depth_per_section'],
                'aspects_covered': [
                    'fundamentals',
                    'edge_cases',
                    'tradeoffs',
                    'alternatives',
                    'examples',
                    'pitfalls',
                    'best_practices'
                ]
            },
            confidence=0.95,  # High confidence - strategy is deterministic
            estimated_improvement=0.55  # +55% depth vs baseline
        )


# Export strategy class
__all__ = ['DeepStrategy']
