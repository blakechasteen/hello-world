#!/usr/bin/env python3
"""
Recursive Convergence Engine - Promptly Integration
===================================================

Enhanced convergence engine with recursive reasoning capabilities.
Extends the base ConvergenceEngine with multi-pass refinement.

Created: 2025-11-15
Integration: Promptly → HoloLoom convergence layer
"""

import logging
from typing import Optional, Callable
from dataclasses import dataclass

from HoloLoom.documentation.types import Query, Spacetime, Features
from HoloLoom.convergence.engine import ConvergenceEngine, CollapseStrategy
from HoloLoom.convergence.recursive_reasoner import (
    BaseRecursiveReasoner,
    create_recursive_reasoner
)
from HoloLoom.protocols.recursive_reasoning import (
    RecursiveConfig,
    ReasoningStrategy,
    ReasoningJournal,
    QualityScoringProtocol
)

logger = logging.getLogger(__name__)


@dataclass
class RecursiveConvergenceResult:
    """
    Result from recursive convergence.

    Extends standard Spacetime with reasoning provenance.
    """
    spacetime: Spacetime
    journal: ReasoningJournal
    iterations: int
    final_quality: float
    strategy_used: ReasoningStrategy


class RecursiveConvergenceEngine(ConvergenceEngine):
    """
    Enhanced convergence engine with recursive reasoning.

    Extends the base ConvergenceEngine to support:
    - Multi-pass refinement loops
    - Quality-driven iteration
    - Scratchpad reasoning provenance
    - Adaptive strategy selection

    Architecture:
    ```
    Query → Features → [DECISION] → Low quality?
                             ↓           ↓ YES
                        Spacetime   [RECURSIVE REFINEMENT]
                                         ↓
                                    Improved Spacetime
    ```
    """

    def __init__(
        self,
        *args,
        enable_recursive: bool = True,
        default_config: Optional[RecursiveConfig] = None,
        quality_scorer: Optional[QualityScoringProtocol] = None,
        **kwargs
    ):
        """
        Initialize recursive convergence engine.

        Args:
            *args: Passed to base ConvergenceEngine
            enable_recursive: Enable recursive reasoning
            default_config: Default recursive configuration
            quality_scorer: Optional quality scoring implementation
            **kwargs: Passed to base ConvergenceEngine
        """
        super().__init__(*args, **kwargs)

        self.enable_recursive = enable_recursive
        self.default_config = default_config or RecursiveConfig(
            strategy=ReasoningStrategy.ADAPTIVE,
            max_iterations=3,
            quality_threshold=0.85,
            min_improvement=0.05
        )
        self.quality_scorer = quality_scorer

        # Will be set when integrated with orchestrator
        self.weaving_fn: Optional[Callable] = None

        logger.info(
            f"RecursiveConvergenceEngine initialized "
            f"(recursive={'enabled' if enable_recursive else 'disabled'})"
        )

    def set_weaving_function(self, weaving_fn: Callable):
        """
        Set the weaving function for recursive refinement.

        This should be WeavingOrchestrator.weave() and is set
        at integration time to avoid circular dependencies.

        Args:
            weaving_fn: async (Query) -> Spacetime
        """
        self.weaving_fn = weaving_fn
        logger.info("Weaving function configured for recursive reasoning")

    async def converge_with_refinement(
        self,
        query: Query,
        features: Features,
        config: Optional[RecursiveConfig] = None
    ) -> RecursiveConvergenceResult:
        """
        Converge to decision with optional recursive refinement.

        This is the main entry point for recursive reasoning:
        1. Initial decision (base convergence)
        2. Quality check
        3. Recursive refinement if needed
        4. Return best result with provenance

        Args:
            query: User's query
            features: Extracted features
            config: Optional custom recursive config

        Returns:
            RecursiveConvergenceResult with spacetime + reasoning journal
        """
        config = config or self.default_config

        # Phase 1: Initial convergence (use base engine)
        initial_spacetime = await self._initial_convergence(query, features)
        initial_quality = initial_spacetime.confidence

        # Check if recursive refinement needed
        needs_refinement = (
            self.enable_recursive and
            self.weaving_fn is not None and
            initial_quality < config.quality_threshold and
            config.strategy != ReasoningStrategy.DIRECT
        )

        if not needs_refinement:
            # Return initial result without refinement
            journal = ReasoningJournal(strategy_used=ReasoningStrategy.DIRECT)
            journal.final_spacetime = initial_spacetime

            return RecursiveConvergenceResult(
                spacetime=initial_spacetime,
                journal=journal,
                iterations=1,
                final_quality=initial_quality,
                strategy_used=ReasoningStrategy.DIRECT
            )

        # Phase 2: Recursive refinement
        logger.info(
            f"Initial quality {initial_quality:.2f} < threshold {config.quality_threshold:.2f}, "
            f"triggering recursive refinement with strategy={config.strategy.value}"
        )

        # Auto-select strategy if ADAPTIVE
        if config.strategy == ReasoningStrategy.ADAPTIVE:
            config.strategy = self._select_strategy(query, initial_quality)
            logger.info(f"Adaptive strategy selected: {config.strategy.value}")

        # Create recursive reasoner
        reasoner = create_recursive_reasoner(
            strategy=config.strategy,
            weaving_fn=self.weaving_fn,
            quality_scorer=self.quality_scorer
        )

        # Execute recursive reasoning
        final_spacetime, journal = await reasoner.reason(
            query=query,
            initial_features=features,
            config=config
        )

        # Return comprehensive result
        return RecursiveConvergenceResult(
            spacetime=final_spacetime,
            journal=journal,
            iterations=len(journal.traces),
            final_quality=journal.get_latest().confidence if journal.traces else initial_quality,
            strategy_used=config.strategy
        )

    async def _initial_convergence(
        self,
        query: Query,
        features: Features
    ) -> Spacetime:
        """
        Perform initial convergence (single-pass decision).

        This would typically call the base ConvergenceEngine logic
        or integrate with WeavingOrchestrator.

        For now, this is a placeholder that would be implemented
        during full integration.
        """
        # TODO: Integrate with actual convergence logic
        # This would call:
        # - Neural core for tool probabilities
        # - Thompson bandit for exploration
        # - Collapse strategy for final decision

        # Placeholder - in real integration, this calls base engine
        return Spacetime(
            response="Initial response (placeholder)",
            confidence=0.75,  # Below threshold to trigger refinement
            metadata={
                "original_query": query.text,
                "features": features
            }
        )

    def _select_strategy(
        self,
        query: Query,
        initial_quality: float
    ) -> ReasoningStrategy:
        """
        Auto-select recursive strategy based on query characteristics.

        Heuristics:
        - Simple queries → REFINE
        - Questions → VERIFY (in HoloLoom's agentic system)
        - Complex multi-part → DECOMPOSE
        - Creative/open-ended → EXPLORE
        - Meta/philosophical → HOFSTADTER

        Args:
            query: User's query
            initial_quality: Quality of initial response

        Returns:
            Selected reasoning strategy
        """
        text_lower = query.text.lower()

        # Meta-questions (consciousness, self-reference)
        meta_keywords = ["consciousness", "self", "meta", "recursive", "paradox"]
        if any(kw in text_lower for kw in meta_keywords):
            return ReasoningStrategy.HOFSTADTER

        # Questions with "why" or "how" (benefit from decomposition)
        if any(q in text_lower for q in ["why", "how does", "explain"]):
            return ReasoningStrategy.DECOMPOSE

        # Creative or open-ended ("best way", "alternatives")
        creative_keywords = ["best way", "alternatives", "options", "creative"]
        if any(kw in text_lower for kw in creative_keywords):
            return ReasoningStrategy.EXPLORE

        # Requests for critique or review
        if any(kw in text_lower for kw in ["review", "critique", "improve"]):
            return ReasoningStrategy.CRITIQUE

        # Default: simple refinement
        return ReasoningStrategy.REFINE

    def get_reasoning_statistics(self) -> dict:
        """
        Get statistics about recursive reasoning usage.

        Returns aggregated metrics:
        - Strategy usage frequency
        - Average iterations per strategy
        - Quality improvements
        - Time spent in recursive reasoning

        Returns:
            Statistics dictionary
        """
        # TODO: Implement statistics tracking
        # This would track:
        # - strategy_counts: Dict[ReasoningStrategy, int]
        # - avg_iterations: Dict[ReasoningStrategy, float]
        # - avg_quality_gain: Dict[ReasoningStrategy, float]

        return {
            "recursive_enabled": self.enable_recursive,
            "default_strategy": self.default_config.strategy.value,
            "quality_threshold": self.default_config.quality_threshold
        }


# Convenience function for creating recursive engine
def create_recursive_engine(
    collapse_strategy: CollapseStrategy = CollapseStrategy.BAYESIAN_BLEND,
    enable_recursive: bool = True,
    quality_threshold: float = 0.85,
    max_iterations: int = 3
) -> RecursiveConvergenceEngine:
    """
    Factory function to create recursive convergence engine.

    Args:
        collapse_strategy: How to collapse probabilities to decisions
        enable_recursive: Enable recursive reasoning
        quality_threshold: Minimum quality to avoid refinement
        max_iterations: Maximum refinement iterations

    Returns:
        Configured RecursiveConvergenceEngine
    """
    config = RecursiveConfig(
        strategy=ReasoningStrategy.ADAPTIVE,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold
    )

    # Note: Base engine parameters would be passed here
    # This is simplified for demonstration
    return RecursiveConvergenceEngine(
        enable_recursive=enable_recursive,
        default_config=config
    )
