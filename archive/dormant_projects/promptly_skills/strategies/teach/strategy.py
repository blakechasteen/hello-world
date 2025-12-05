"""Teach Strategy - Few-shot Edge Case Learning"""

from pathlib import Path
import yaml

from HoloLoom.prompting.strategy import (
    TemplateStrategy,
    StrategyContext,
    StrategyResult,
    StrategyCategory
)


class TeachStrategy(TemplateStrategy):
    def __init__(self):
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self._full_config = yaml.safe_load(f)

        template_path = Path(__file__).parent / "template.md"
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        super().__init__(template, self._full_config['config'])

    @property
    def name(self) -> str:
        return "teach"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.META_PROMPTING

    @property
    def description(self) -> str:
        return "Few-shot Edge Case Learning - Show boundary condition examples"

    def can_apply(self, context: StrategyContext) -> float:
        query_lower = context.query.lower()
        detection_config = self._full_config['detection']

        confidence = detection_config['base_confidence']

        high_keywords = detection_config['high_confidence_keywords']
        if any(keyword in query_lower for keyword in high_keywords):
            confidence += detection_config['keyword_boost']

        medium_keywords = detection_config.get('medium_confidence_keywords', [])
        if any(keyword in query_lower for keyword in medium_keywords):
            confidence += detection_config['keyword_boost'] * 0.6

        return max(0.0, min(1.0, confidence))

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        enhanced_query = self.template.format(query=context.query)

        return StrategyResult(
            enhanced_query=enhanced_query,
            metadata={
                'strategy': 'teach',
                'min_examples': self._full_config['config']['min_examples'],
                'example_types': self._full_config['config']['example_types']
            },
            confidence=0.90,
            estimated_improvement=0.50
        )


__all__ = ['TeachStrategy']
