"""
Composite Strategy for Chaining

Allows multiple strategies to be composed into a pipeline.
Example: verify+challenge chains verification with adversarial testing.
"""

from typing import List
import logging

from .strategy import (
    PromptingStrategy,
    StrategyContext,
    StrategyResult,
    StrategyCategory
)

logger = logging.getLogger(__name__)


class CompositeStrategy(PromptingStrategy):
    """
    Composes multiple strategies into a sequential pipeline.

    Each strategy in the pipeline receives the output of the previous
    strategy as its input. This enables powerful combinations like:

    - verify+challenge: Verify output then adversarially test it
    - optimize+deep: Optimize prompt then force exhaustive depth
    - scaffold+prime: Apply template then prime with quality examples

    Example:
        >>> verify = VerifyStrategy()
        >>> challenge = ChallengeStrategy()
        >>> combined = CompositeStrategy([verify, challenge])
        >>> # Or use shorthand:
        >>> combined = verify + challenge
    """

    def __init__(self, strategies: List[PromptingStrategy]):
        """
        Initialize composite strategy.

        Args:
            strategies: List of strategies to chain together

        Raises:
            ValueError: If strategies list is empty
        """
        if not strategies:
            raise ValueError("CompositeStrategy requires at least one strategy")

        self.strategies = strategies
        logger.debug(f"Created composite: {' + '.join(s.name for s in strategies)}")

    @property
    def name(self) -> str:
        """Composite name is '+'-joined strategy names"""
        return "+".join(s.name for s in self.strategies)

    @property
    def category(self) -> StrategyCategory:
        """Category is always COMPOSITE"""
        return StrategyCategory.COMPOSITE

    @property
    def description(self) -> str:
        """Description lists all chained strategies"""
        strategy_names = [s.name for s in self.strategies]
        return f"Composite pipeline: {' → '.join(strategy_names)}"

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        """
        Apply strategies sequentially.

        Each strategy receives the enhanced query from the previous
        strategy. Metadata from all strategies is accumulated.

        Args:
            context: Initial query context

        Returns:
            StrategyResult with final enhanced query and accumulated metadata

        Example:
            >>> composite = verify + challenge
            >>> context = StrategyContext(query="review security")
            >>> result = await composite.enhance(context)
            >>> print(result.metadata['pipeline'])
            ['verify', 'challenge']
        """
        current_context = context
        pipeline_metadata = []
        pipeline_confidence = 1.0

        logger.info(f"Executing composite pipeline: {self.name}")

        for i, strategy in enumerate(self.strategies):
            logger.debug(f"Pipeline step {i+1}/{len(self.strategies)}: {strategy.name}")

            # Apply strategy
            result = await strategy.enhance(current_context)

            # Accumulate metadata
            pipeline_metadata.append({
                'strategy': strategy.name,
                'category': strategy.category.value,
                'metadata': result.metadata,
                'confidence': result.confidence
            })

            # Update confidence (product of all confidences)
            pipeline_confidence *= result.confidence

            # Update context for next strategy
            current_context = current_context.with_query(result.enhanced_query)
            current_context.previous_result = result

        logger.info(f"Pipeline complete: {self.name}, confidence={pipeline_confidence:.2f}")

        # Final result
        return StrategyResult(
            enhanced_query=current_context.query,
            metadata={
                'pipeline': [s.name for s in self.strategies],
                'steps': pipeline_metadata,
                'composite': True
            },
            confidence=pipeline_confidence,
            requires_followup=False
        )

    def can_apply(self, context: StrategyContext) -> float:
        """
        Composite applicability is the minimum of all strategies.

        A chain is only as strong as its weakest link. If any strategy
        in the pipeline is not applicable, the whole pipeline is less applicable.

        Args:
            context: Query context

        Returns:
            Minimum confidence of all strategies in pipeline

        Example:
            >>> composite = verify + challenge
            >>> context = StrategyContext(query="analyze")
            >>> confidence = composite.can_apply(context)
            >>> # Returns min(verify.can_apply(), challenge.can_apply())
        """
        if not self.strategies:
            return 0.0

        confidences = [s.can_apply(context) for s in self.strategies]
        min_confidence = min(confidences)

        logger.debug(
            f"Composite '{self.name}' applicability: "
            f"{min_confidence:.2f} (min of {confidences})"
        )

        return min_confidence

    def append(self, strategy: PromptingStrategy) -> 'CompositeStrategy':
        """
        Append another strategy to the pipeline.

        Args:
            strategy: Strategy to append

        Returns:
            New composite with appended strategy

        Example:
            >>> composite = verify + challenge
            >>> extended = composite.append(deep)
            >>> # Now: verify + challenge + deep
        """
        return CompositeStrategy(self.strategies + [strategy])

    def prepend(self, strategy: PromptingStrategy) -> 'CompositeStrategy':
        """
        Prepend another strategy to the pipeline.

        Args:
            strategy: Strategy to prepend

        Returns:
            New composite with prepended strategy

        Example:
            >>> composite = verify + challenge
            >>> extended = composite.prepend(optimize)
            >>> # Now: optimize + verify + challenge
        """
        return CompositeStrategy([strategy] + self.strategies)

    def __add__(self, other: PromptingStrategy) -> 'CompositeStrategy':
        """
        Syntactic sugar for appending: composite + strategy

        Example:
            >>> pipeline = verify + challenge + deep
        """
        return self.append(other)

    def __len__(self) -> int:
        """Return number of strategies in pipeline"""
        return len(self.strategies)

    def __iter__(self):
        """Iterate over strategies in pipeline"""
        return iter(self.strategies)

    def __repr__(self) -> str:
        return f"CompositeStrategy({self.name})"


def create_pipeline(strategy_names: List[str]) -> CompositeStrategy:
    """
    Create composite pipeline from strategy names.

    Args:
        strategy_names: List of strategy names to chain

    Returns:
        CompositeStrategy with loaded strategies

    Raises:
        ValueError: If any strategy name is unknown

    Example:
        >>> pipeline = create_pipeline(['verify', 'challenge', 'deep'])
        >>> # Equivalent to: verify + challenge + deep
    """
    from .registry import get_strategy

    strategies = []
    for name in strategy_names:
        strategy = get_strategy(name)
        if strategy is None:
            raise ValueError(f"Unknown strategy: {name}")
        strategies.append(strategy)

    return CompositeStrategy(strategies)


def parse_pipeline(pipeline_str: str) -> CompositeStrategy:
    """
    Parse pipeline string into CompositeStrategy.

    Args:
        pipeline_str: Pipeline string like "verify+challenge+deep"

    Returns:
        CompositeStrategy

    Raises:
        ValueError: If any strategy name is unknown

    Example:
        >>> pipeline = parse_pipeline("verify+challenge")
        >>> print(pipeline.name)
        "verify+challenge"
    """
    strategy_names = [name.strip() for name in pipeline_str.split('+')]
    return create_pipeline(strategy_names)
