"""MetaChain Strategy - Intelligent Strategy Chaining

Automatically detects query type and chains appropriate strategies.
Showcases the elegant composability of the framework.
"""

from pathlib import Path
from typing import Dict, List, Any
import yaml

from HoloLoom.prompting.strategy import (
    PromptingStrategy,
    StrategyContext,
    StrategyResult,
    StrategyCategory
)


class MetaChainStrategy(PromptingStrategy):
    """
    Meta-Chain Prompting strategy.

    Intelligently chains multiple strategies based on query characteristics:
    - Learning queries → deep + teach + verify
    - Problem solving → scaffold + teach + verify
    - Quality focus → prime + optimize + challenge
    - Tradeoff analysis → debate + deep + prime
    - Uncertain queries → temp_sim + scaffold + verify

    This strategy showcases the elegant composability of the framework.

    Example:
        >>> strategy = MetaChainStrategy()
        >>> context = StrategyContext(query="explain neural networks thoroughly")
        >>> result = await strategy.enhance(context)
        >>> # Automatically chains: deep + teach + verify
    """

    def __init__(self):
        """Initialize meta-chain strategy with config and template."""
        # Load config
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self._full_config = yaml.safe_load(f)

        # Load template
        template_path = Path(__file__).parent / "template.md"
        with open(template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()

    @property
    def name(self) -> str:
        """Strategy name."""
        return "meta_chain"

    @property
    def category(self) -> StrategyCategory:
        """Strategy category."""
        return StrategyCategory.META_PROMPTING

    @property
    def description(self) -> str:
        """Strategy description."""
        return "Meta-Chain Prompting - Intelligently chain multiple strategies"

    def _detect_query_type(self, query: str) -> str:
        """
        Detect query type from keywords.

        Args:
            query: User query

        Returns:
            Query type (learning_query, problem_solving, quality_focus, etc.)
        """
        query_lower = query.lower()
        query_type_keywords = self._full_config['detection']['query_type_keywords']

        # Check each query type
        for query_type, keywords in query_type_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return f"{query_type}_query" if query_type != 'uncertain' else "uncertain_query"

        # Default to learning query
        return "learning_query"

    def can_apply(self, context: StrategyContext) -> float:
        """
        Determine if this strategy is appropriate for the context.

        Medium-high confidence for most queries (meta-chain is versatile).

        Args:
            context: Strategy context with query

        Returns:
            Confidence score (0-1)
        """
        query_lower = context.query.lower()
        detection_config = self._full_config['detection']

        # Start with base confidence
        confidence = detection_config['base_confidence']

        # Check for explicit meta-chain keywords
        high_keywords = detection_config['high_confidence_keywords']
        if any(keyword in query_lower for keyword in high_keywords):
            confidence += detection_config['keyword_boost']

        # Boost if query matches any query type (versatile strategy)
        query_type = self._detect_query_type(context.query)
        if query_type in self._full_config['config']['chaining_rules']:
            confidence += 0.25  # Medium boost for recognized query types

        # Ensure in valid range
        return max(0.0, min(1.0, confidence))

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        """
        Enhance query by chaining appropriate strategies.

        Args:
            context: Strategy context with query

        Returns:
            Strategy result with chained strategy instructions
        """
        # Detect query type
        query_type = self._detect_query_type(context.query)

        # Get chaining rule
        chaining_rules = self._full_config['config']['chaining_rules']
        rule = chaining_rules.get(query_type, chaining_rules['learning_query'])

        # Format template
        chain_display = ' → '.join(rule['chain'])
        strategy_instructions = self._generate_strategy_instructions(rule['chain'])
        quality_criteria = self._generate_quality_criteria(rule['chain'])

        enhanced_query = self.template.format(
            query=context.query,
            query_type=query_type.replace('_', ' ').title(),
            chain=' + '.join(rule['chain']),
            rationale=rule['rationale'],
            chain_display=chain_display,
            strategy_instructions=strategy_instructions,
            quality_criteria=quality_criteria
        )

        # Return result with metadata
        return StrategyResult(
            enhanced_query=enhanced_query,
            metadata={
                'strategy': 'meta_chain',
                'query_type': query_type,
                'chained_strategies': rule['chain'],
                'chain_length': len(rule['chain']),
                'rationale': rule['rationale']
            },
            confidence=0.88,  # High confidence for intelligent chaining
            estimated_improvement=0.65  # +65% through multi-strategy
        )

    def _generate_strategy_instructions(self, chain: List[str]) -> str:
        """Generate instructions for each strategy in chain."""
        instructions = []
        for i, strategy_name in enumerate(chain, 1):
            instructions.append(f"**Stage {i}: {strategy_name.upper()}**")
            instructions.append(self._get_strategy_description(strategy_name))
            instructions.append("")

        return "\n".join(instructions)

    def _generate_quality_criteria(self, chain: List[str]) -> str:
        """Generate quality criteria for all strategies in chain."""
        criteria = []
        for strategy_name in chain:
            criteria.append(f"✓ **{strategy_name}**: {self._get_strategy_criterion(strategy_name)}")

        return "\n".join(criteria)

    def _get_strategy_description(self, strategy_name: str) -> str:
        """Get description for a strategy."""
        descriptions = {
            'deep': 'Provide exhaustive depth covering fundamentals, edge cases, tradeoffs, alternatives, examples, pitfalls, and best practices.',
            'teach': 'Show concrete examples: typical case, edge case, and error case.',
            'verify': 'Verify completeness and accuracy through systematic checking.',
            'scaffold': 'Use structured reasoning template: understand, decompose, analyze, synthesize, validate, conclude.',
            'prime': 'Meet world-class quality standards: clarity, completeness, accuracy, elegance, practicality.',
            'optimize': 'Refine through 3 iterations: add constraints, resolve ambiguities, enhance reasoning.',
            'challenge': 'Find problems through adversarial thinking: identify 5+ vulnerabilities with exploitation scenarios.',
            'debate': 'Explore through multi-persona debate: optimist, skeptic, pragmatist perspectives.',
            'temp_sim': 'Show responses at different confidence levels: high, medium, low certainty.'
        }
        return descriptions.get(strategy_name, 'Apply this strategy to enhance the response.')

    def _get_strategy_criterion(self, strategy_name: str) -> str:
        """Get quality criterion for a strategy."""
        criteria = {
            'deep': '7 sections with 3+ points each',
            'teach': '3 examples (typical, edge, error)',
            'verify': 'Verified for accuracy and completeness',
            'scaffold': '6-step reasoning shown',
            'prime': 'World-class quality on 5 dimensions',
            'optimize': '3 iterations of refinement',
            'challenge': '5+ problems with exploits',
            'debate': '3 personas with synthesis',
            'temp_sim': '3 confidence levels shown'
        }
        return criteria.get(strategy_name, 'Strategy requirements met')


# Export strategy class
__all__ = ['MetaChainStrategy']
