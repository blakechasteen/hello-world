"""
Scaffold Strategy - Zero-shot Chain-of-Thought Structure

Provides structured reasoning template with blanks that must be filled in.
Forces explicit step-by-step thinking: understand, decompose, analyze, synthesize, validate, conclude.
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


class ScaffoldStrategy(TemplateStrategy):
    """
    Zero-shot CoT Scaffolding strategy.

    Provides structured reasoning template that forces explicit thinking:
    1. Understand the problem
    2. Decompose into subproblems
    3. Analyze each subproblem
    4. Synthesize insights
    5. Validate reasoning
    6. State conclusion

    All blanks must be filled in - no skipping steps.

    Auto-detects:
    - High confidence (0.85) for "step by step", "show your work"
    - Medium confidence (0.65) for problem-solving queries
    - Low confidence for simple factual queries

    Example:
        >>> strategy = ScaffoldStrategy()
        >>> context = StrategyContext(query="solve this math problem step by step")
        >>> result = await strategy.enhance(context)
        >>> # Result includes 6-step reasoning template
    """

    def __init__(self):
        """Initialize scaffold strategy with config and template."""
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
        return "scaffold"

    @property
    def category(self) -> StrategyCategory:
        """Strategy category."""
        return StrategyCategory.SELF_CORRECTION

    @property
    def description(self) -> str:
        """Strategy description."""
        return "Zero-shot CoT Structure - Template with reasoning blanks"

    def can_apply(self, context: StrategyContext) -> float:
        """
        Determine if this strategy is appropriate for the context.

        High confidence (0.85+) for explicit step-by-step requests.
        Medium confidence (0.65+) for problem-solving queries.
        Low confidence for simple factual queries.

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

        # Check for medium confidence keywords
        elif any(keyword in query_lower for keyword in detection_config['medium_confidence_keywords']):
            confidence += detection_config['keyword_boost'] * 0.7  # Slightly lower

        # Additional boost for problem-solving indicators
        problem_indicators = ['solve', 'calculate', 'find', 'determine', 'prove']
        if any(indicator in query_lower for indicator in problem_indicators):
            confidence += detection_config['problem_solving_boost']

        # Additional boost for "how" and "why" questions (need reasoning)
        if query_lower.startswith('how ') or query_lower.startswith('why '):
            confidence += 0.15

        # Penalty for very short queries (likely don't need scaffolding)
        if len(context.query.split()) < 5:
            confidence -= 0.1

        # Ensure in valid range
        return max(0.0, min(1.0, confidence))

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        """
        Enhance query with structured reasoning template.

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
                'strategy': 'scaffold',
                'reasoning_steps': 6,
                'steps': [
                    'understand',
                    'decompose',
                    'analyze',
                    'synthesize',
                    'validate',
                    'conclude'
                ],
                'require_thinking_shown': self._full_config['config']['require_thinking_shown'],
                'min_reasoning_steps': self._full_config['config']['min_reasoning_steps']
            },
            confidence=0.92,  # High confidence - structured template works well
            estimated_improvement=0.42  # +42% reasoning clarity
        )


# Export strategy class
__all__ = ['ScaffoldStrategy']
