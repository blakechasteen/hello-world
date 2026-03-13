"""
Refinement Strategies - Intelligent Strategy Selection
=======================================================

Selects optimal refinement strategies based on query characteristics and learns
from outcomes using Thompson Sampling.

Created: 2025-11-20
Author: HoloLoom Team
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from hololoom.bandits.beta_arm import BetaArm

from hololoom.protocols.recursive_reasoning import (
    RefinementStrategy,
    RefinementStep,
    StrategySelectorProtocol
)

logger = logging.getLogger(__name__)


# ============================================================================
# Query Classification
# ============================================================================

class QueryClassifier:
    """
    Classify queries into types for strategy selection.

    Query Types:
    - factual: "What is X?"
    - procedural: "How to do X?"
    - analytical: "Why does X happen?"
    - comparative: "Compare X and Y"
    - creative: Open-ended, subjective
    - technical: Code, math, algorithms
    """

    def __init__(self):
        self.factual_keywords = ["what is", "define", "explain", "describe"]
        self.procedural_keywords = ["how to", "how do", "steps", "process", "method"]
        self.analytical_keywords = ["why", "reason", "cause", "analyze", "evaluate"]
        self.comparative_keywords = ["compare", "difference", "vs", "versus", "better"]
        self.technical_keywords = ["algorithm", "code", "function", "implement", "optimize"]

    def classify(self, query: str) -> str:
        """
        Classify query type.

        Args:
            query: Query text

        Returns:
            Query type string
        """
        query_lower = query.lower()

        # Check patterns
        if any(kw in query_lower for kw in self.factual_keywords):
            return "factual"
        elif any(kw in query_lower for kw in self.procedural_keywords):
            return "procedural"
        elif any(kw in query_lower for kw in self.analytical_keywords):
            return "analytical"
        elif any(kw in query_lower for kw in self.comparative_keywords):
            return "comparative"
        elif any(kw in query_lower for kw in self.technical_keywords):
            return "technical"
        else:
            return "creative"


# ============================================================================
# Strategy Performance Tracker
# ============================================================================

@dataclass
class StrategyStats:
    """Statistics for a refinement strategy.

    Delegates Beta math to BetaArm (canonical implementation).
    """
    strategy: RefinementStrategy
    total_uses: int = 0
    successful_uses: int = 0  # Improvement > 0
    total_improvement: float = 0.0
    avg_improvement: float = 0.0
    success_rate: float = 0.0

    # Thompson Sampling via BetaArm
    _arm: BetaArm = field(default_factory=BetaArm)

    @property
    def alpha(self) -> float:
        return self._arm.alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._arm.alpha = value

    @property
    def beta(self) -> float:
        return self._arm.beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._arm.beta = value

    def update(self, improvement: float):
        """Update stats with new outcome."""
        self.total_uses += 1
        self.total_improvement += improvement

        if improvement > 0:
            self.successful_uses += 1
            self._arm.update(success=True, weight=improvement)
        else:
            self._arm.update(success=False, weight=abs(improvement) if improvement < 0 else 0.01)

        # Recalculate averages
        self.avg_improvement = self.total_improvement / self.total_uses
        self.success_rate = self.successful_uses / self.total_uses

    def get_thompson_sample(self) -> float:
        """Sample from Beta distribution for Thompson Sampling."""
        return self._arm.sample()

    def get_expected_reward(self) -> float:
        """Get expected reward (mean of Beta distribution)."""
        return self._arm.expected_value()


# ============================================================================
# Strategy Selector
# ============================================================================

class StrategySelector:
    """
    Select optimal refinement strategy using Thompson Sampling.

    Learning Process:
    1. Track performance of each strategy per query type
    2. Use Thompson Sampling to balance exploration/exploitation
    3. Adapt strategy selection based on observed improvements
    """

    def __init__(self, enable_learning: bool = True):
        """
        Initialize strategy selector.

        Args:
            enable_learning: Whether to learn from outcomes
        """
        self.enable_learning = enable_learning
        self.classifier = QueryClassifier()

        # Strategy performance tracking
        # Map: (query_type, strategy) → StrategyStats
        self.strategy_stats: Dict[Tuple[str, RefinementStrategy], StrategyStats] = {}

        # Initialize all combinations
        for query_type in ["factual", "procedural", "analytical", "comparative", "technical", "creative"]:
            for strategy in RefinementStrategy:
                key = (query_type, strategy)
                self.strategy_stats[key] = StrategyStats(strategy=strategy)

        logger.info(f"StrategySelector initialized (learning={'on' if enable_learning else 'off'})")

    async def select_strategy(
        self,
        query: str,
        current_confidence: float,
        refinement_history: List[RefinementStep]
    ) -> RefinementStrategy:
        """
        Select optimal refinement strategy.

        Args:
            query: Current query
            current_confidence: Current confidence score
            refinement_history: History of refinements

        Returns:
            Selected refinement strategy
        """
        # Classify query
        query_type = self.classifier.classify(query)
        logger.debug(f"Query classified as: {query_type}")

        # Use Thompson Sampling to select strategy
        if self.enable_learning and len(refinement_history) > 0:
            strategy = self._thompson_sample_strategy(query_type, current_confidence)
        else:
            strategy = self._heuristic_strategy(query_type, current_confidence)

        logger.info(f"Selected strategy: {strategy.value} (query_type={query_type}, confidence={current_confidence:.2f})")
        return strategy

    def _thompson_sample_strategy(self, query_type: str, confidence: float) -> RefinementStrategy:
        """Select strategy using Thompson Sampling."""
        # Sample from each strategy's Beta distribution
        samples = {}
        for strategy in RefinementStrategy:
            key = (query_type, strategy)
            if key in self.strategy_stats:
                samples[strategy] = self.strategy_stats[key].get_thompson_sample()
            else:
                samples[strategy] = np.random.beta(1.0, 1.0)  # Uniform prior

        # Pick strategy with highest sample
        best_strategy = max(samples.items(), key=lambda x: x[1])[0]

        # Inject some entropy (5% chance of random)
        if np.random.rand() < 0.05:
            best_strategy = np.random.choice(list(RefinementStrategy))
            logger.debug("Random strategy injection (5% entropy)")

        return best_strategy

    def _heuristic_strategy(self, query_type: str, confidence: float) -> RefinementStrategy:
        """Select strategy using heuristics (fallback if no learning)."""
        # Low confidence → Need more information
        if confidence < 0.5:
            if query_type in ["factual", "technical"]:
                return RefinementStrategy.EXPAND_SEARCH
            else:
                return RefinementStrategy.DECOMPOSE

        # Medium confidence → Need verification or alternate perspective
        elif confidence < 0.7:
            if query_type == "analytical":
                return RefinementStrategy.VERIFY_AND_CORRECT
            elif query_type == "comparative":
                return RefinementStrategy.MULTI_PERSPECTIVE
            else:
                return RefinementStrategy.RERANK

        # High confidence but not perfect → Polish
        else:
            if query_type == "procedural":
                return RefinementStrategy.VERIFY_AND_CORRECT
            else:
                return RefinementStrategy.ALTERNATE_MODE

    async def learn_from_outcome(
        self,
        strategy: RefinementStrategy,
        improvement: float,
        query_type: str
    ):
        """
        Learn from refinement outcome.

        Args:
            strategy: Strategy that was used
            improvement: Quality improvement achieved
            query_type: Type of query
        """
        if not self.enable_learning:
            return

        key = (query_type, strategy)

        if key not in self.strategy_stats:
            self.strategy_stats[key] = StrategyStats(strategy=strategy)

        self.strategy_stats[key].update(improvement)

        logger.info(
            f"Learned from outcome: strategy={strategy.value}, improvement={improvement:+.3f}, "
            f"query_type={query_type}, success_rate={self.strategy_stats[key].success_rate:.2f}"
        )

    def get_strategy_statistics(self, query_type: Optional[str] = None) -> Dict[str, any]:
        """
        Get strategy performance statistics.

        Args:
            query_type: Optional filter by query type

        Returns:
            Dictionary of statistics
        """
        if query_type:
            # Filter by query type
            relevant_stats = {
                strategy.value: self.strategy_stats[(query_type, strategy)]
                for strategy in RefinementStrategy
                if (query_type, strategy) in self.strategy_stats
            }
        else:
            # Aggregate across all query types
            aggregated: Dict[str, List[StrategyStats]] = defaultdict(list)
            for (qtype, strategy), stats in self.strategy_stats.items():
                aggregated[strategy.value].append(stats)

            relevant_stats = {}
            for strategy_name, stats_list in aggregated.items():
                if stats_list:
                    total_uses = sum(s.total_uses for s in stats_list)
                    if total_uses > 0:
                        relevant_stats[strategy_name] = {
                            "total_uses": total_uses,
                            "avg_improvement": sum(s.total_improvement for s in stats_list) / total_uses,
                            "success_rate": sum(s.successful_uses for s in stats_list) / total_uses,
                            "expected_reward": np.mean([s.get_expected_reward() for s in stats_list])
                        }

        return {
            "query_type": query_type or "all",
            "strategies": relevant_stats
        }

    def get_best_strategy_for_type(self, query_type: str) -> Tuple[RefinementStrategy, float]:
        """
        Get best performing strategy for a query type.

        Args:
            query_type: Query type

        Returns:
            (best_strategy, expected_reward) tuple
        """
        best_strategy = None
        best_reward = 0.0

        for strategy in RefinementStrategy:
            key = (query_type, strategy)
            if key in self.strategy_stats:
                reward = self.strategy_stats[key].get_expected_reward()
                if reward > best_reward:
                    best_reward = reward
                    best_strategy = strategy

        return best_strategy or RefinementStrategy.EXPAND_SEARCH, best_reward


# ============================================================================
# Strategy Executor (Maps strategy to execution)
# ============================================================================

class StrategyExecutor:
    """
    Execute refinement strategies.

    Each strategy translates to specific actions in the RAG system.
    """

    @staticmethod
    def get_execution_params(strategy: RefinementStrategy) -> Dict[str, any]:
        """
        Get execution parameters for strategy.

        Args:
            strategy: Refinement strategy

        Returns:
            Dictionary of execution parameters
        """
        if strategy == RefinementStrategy.EXPAND_SEARCH:
            return {
                "max_sources": 10,  # Expand from default 5
                "mode": "research",  # More thorough
                "enable_reranking": True
            }

        elif strategy == RefinementStrategy.RERANK:
            return {
                "max_sources": 5,
                "mode": "verify",
                "enable_reranking": True  # Force reranking
            }

        elif strategy == RefinementStrategy.ALTERNATE_MODE:
            return {
                "max_sources": 5,
                "mode": "plan_execute",  # Different reasoning mode
                "enable_reranking": False
            }

        elif strategy == RefinementStrategy.DECOMPOSE:
            return {
                "decompose": True,  # Trigger decomposition
                "max_sources": 3,   # Per sub-query
                "mode": "direct"
            }

        elif strategy == RefinementStrategy.VERIFY_AND_CORRECT:
            return {
                "max_sources": 7,
                "mode": "verify",
                "enable_reranking": True,
                "cross_check": True  # Multiple verification passes
            }

        elif strategy == RefinementStrategy.MULTI_PERSPECTIVE:
            return {
                "max_sources": 8,
                "mode": "research",
                "perspective_count": 3  # Generate 3 perspectives
            }

        else:
            return {"max_sources": 5, "mode": "verify"}


# ============================================================================
# Factory
# ============================================================================

def create_strategy_selector(enable_learning: bool = True) -> StrategySelector:
    """
    Create strategy selector.

    Args:
        enable_learning: Whether to enable learning from outcomes

    Returns:
        Configured StrategySelector
    """
    return StrategySelector(enable_learning=enable_learning)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("="*80)
        print("Refinement Strategy Selection Demo")
        print("="*80 + "\n")

        selector = StrategySelector(enable_learning=True)

        # Simulate query refinement scenarios
        scenarios = [
            ("What is Thompson Sampling?", 0.65, "factual"),
            ("How to implement gradient descent?", 0.55, "procedural"),
            ("Why does overfitting occur in neural networks?", 0.70, "analytical"),
            ("Compare SVD and PCA dimensionality reduction", 0.60, "comparative"),
        ]

        print("Testing strategy selection:\n")

        for query, confidence, expected_type in scenarios:
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Confidence: {confidence:.2f}")
            print(f"{'='*80}")

            # Select strategy
            strategy = await selector.select_strategy(query, confidence, [])

            print(f"  Selected: {strategy.value}")
            print(f"  Expected type: {expected_type}")

            # Get execution params
            params = StrategyExecutor.get_execution_params(strategy)
            print(f"  Execution params: {params}")

            # Simulate outcome
            simulated_improvement = np.random.uniform(0.05, 0.20)
            await selector.learn_from_outcome(strategy, simulated_improvement, expected_type)

            print(f"  Simulated improvement: {simulated_improvement:+.3f}\n")

        # Show learned statistics
        print("\n" + "="*80)
        print("Learned Strategy Statistics:")
        print("="*80)

        for query_type in ["factual", "procedural", "analytical", "comparative"]:
            stats = selector.get_strategy_statistics(query_type=query_type)
            print(f"\n{query_type.upper()}:")

            if stats["strategies"]:
                for strategy, s in stats["strategies"].items():
                    if isinstance(s, dict):
                        print(f"  {strategy:20s} Uses={s['total_uses']:2d}, "
                              f"Avg Improvement={s['avg_improvement']:+.3f}, "
                              f"Success Rate={s['success_rate']:.2f}")

        print("\n✓ Demo complete!")

    asyncio.run(demo())
