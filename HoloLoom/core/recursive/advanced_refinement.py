"""
Advanced Recursive Refinement - Phase 4 (Enhanced with UnifiedMRF)
====================================================================
Multiple refinement strategies, quality trajectory tracking, and learning from
successful refinements.

**UPDATED (Phase 1.2 - Nov 2025):** Integrated with UnifiedMRF for enhanced prompting.

Refinement Strategies:
- REFINE: Iteratively expand and improve query
- CRITIQUE: Self-critique result and regenerate
- VERIFY: Multi-pass cross-check against multiple sources (accuracy→completeness→consistency)
- ELEGANCE: Iteratively polish for clarity, simplicity, and beauty
- HOFSTADTER: Strange loop self-reference for deep reasoning

This module enhances Phase 1's basic refinement with sophisticated strategies
that learn from successful refinement patterns.

Multi-Pass Philosophy:
The ELEGANCE and VERIFY strategies embrace multiple passes:
- Each pass improves a specific dimension (clarity, accuracy, simplicity)
- Quality trajectory shows incremental improvement
- Learning identifies when additional passes yield diminishing returns

Integration with UnifiedMRF:
- Refinement prompts use 7-component metaprompt framework
- Model adapters optimize for Claude/Gemini/GPT/Ollama
- Quality metrics track improvement trajectory
"""

import asyncio
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Callable, Union
from datetime import datetime

from HoloLoom.protocols.types import Query
from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.recursive.scratchpad import Scratchpad, ScratchpadEntry
from HoloLoom.prompting.unified_mrf import (
    UnifiedMRF,
    RefinementStrategyType,
    ModelProvider
)


class RefinementStrategy(Enum):
    """
    Available refinement strategies.

    .. deprecated:: 1.1.0
        `RefinementStrategy` is deprecated and will be removed in v2.0.0.
        Use `RefinementStrategyType` from `HoloLoom.prompting.unified_mrf` instead.

        Migration:
            # Old (deprecated)
            from HoloLoom.recursive.advanced_refinement import RefinementStrategy
            strategy = RefinementStrategy.REFINE

            # New (recommended)
            from HoloLoom.prompting.unified_mrf import RefinementStrategyType
            strategy = RefinementStrategyType.REFINE

    **DEPRECATED (Phase 1.4 - Nov 2025):** This enum conflicts with
    `HoloLoom.writing.core.protocol.RefinementStrategy`. To resolve naming collision,
    use `RefinementStrategyType` from UnifiedMRF instead.

    This enum is kept for backward compatibility with existing code.
    Internally, we map to RefinementStrategyType from UnifiedMRF.
    """
    REFINE = "refine"          # Iterative expansion and improvement
    CRITIQUE = "critique"      # Self-critique and regenerate
    VERIFY = "verify"          # Multi-pass cross-check multiple sources
    ELEGANCE = "elegance"      # Multi-pass polish for clarity and simplicity
    HOFSTADTER = "hofstadter"  # Strange loop self-reference

    def __init__(self, value):
        """Emit deprecation warning on initialization."""
        warnings.warn(
            "RefinementStrategy from HoloLoom.recursive.advanced_refinement is deprecated. "
            "Use RefinementStrategyType from HoloLoom.prompting.unified_mrf instead. "
            "This enum will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=3
        )

    @classmethod
    def to_unified_type(cls, strategy: 'RefinementStrategy') -> RefinementStrategyType:
        """
        Convert to UnifiedMRF RefinementStrategyType.

        .. deprecated:: 1.1.0
            Use `RefinementStrategyType` directly instead of converting.
        """
        mapping = {
            cls.REFINE: RefinementStrategyType.REFINE,
            cls.CRITIQUE: RefinementStrategyType.CRITIQUE,
            cls.VERIFY: RefinementStrategyType.VERIFY,
            cls.ELEGANCE: RefinementStrategyType.ELEGANCE,
            cls.HOFSTADTER: RefinementStrategyType.HOFSTADTER,
        }
        return mapping[strategy]


@dataclass
class QualityMetrics:
    """Metrics tracking quality improvement trajectory"""
    confidence: float
    threads_activated: int
    motifs_detected: int
    context_size: int
    response_length: int
    timestamp: datetime = field(default_factory=datetime.now)

    def score(self) -> float:
        """
        Composite quality score combining multiple factors.

        Returns:
            Quality score [0.0, 1.0]
        """
        # Confidence is primary (70% weight)
        # Context richness is secondary (20% weight)
        # Response completeness is tertiary (10% weight)
        context_richness = min(1.0, (self.threads_activated + self.motifs_detected) / 10.0)
        response_completeness = min(1.0, self.response_length / 500.0)

        return (
            0.7 * self.confidence +
            0.2 * context_richness +
            0.1 * response_completeness
        )


@dataclass
class RefinementResult:
    """
    Result of refinement process with full trajectory.

    **UPDATED (Phase 1.4):** strategy_used accepts both RefinementStrategy and RefinementStrategyType.
    """
    final_spacetime: Spacetime
    trajectory: List[QualityMetrics]
    strategy_used: Union[RefinementStrategy, RefinementStrategyType]
    iterations: int
    improved: bool
    improvement_rate: float  # Quality improvement per iteration

    def summary(self) -> str:
        """Generate human-readable summary"""
        initial_quality = self.trajectory[0].score()
        final_quality = self.trajectory[-1].score()

        return (
            f"Refinement Summary:\n"
            f"  Strategy: {self.strategy_used.value}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Quality: {initial_quality:.2f} → {final_quality:.2f}\n"
            f"  Improvement: {'+' if self.improved else ''}{final_quality - initial_quality:.3f}\n"
            f"  Rate: {self.improvement_rate:.3f}/iteration"
        )


@dataclass
class RefinementPattern:
    """Learned pattern from successful refinement"""
    strategy: RefinementStrategy
    initial_quality: float
    final_quality: float
    iterations: int
    query_characteristics: Dict[str, Any]  # Length, motifs, etc.
    improvement_rate: float
    occurrences: int = 1
    avg_improvement: float = 0.0

    def update(self, improvement: float):
        """Update pattern with new occurrence"""
        self.occurrences += 1
        self.avg_improvement = (
            (self.avg_improvement * (self.occurrences - 1) + improvement) /
            self.occurrences
        )


class AdvancedRefiner:
    """
    Enhanced recursive refiner with multiple strategies and learning.

    **UPDATED (Phase 1.2):** Integrated with UnifiedMRF for enhanced prompting.

    This refiner can:
    1. Apply different refinement strategies based on query type
    2. Track quality trajectories across iterations
    3. Learn which strategies work best for which queries
    4. Adapt strategy selection based on past successes
    5. Use 7-component metaprompt framework for all refinements
    6. Apply model-specific optimizations (Claude/Gemini/GPT/Ollama)
    """

    def __init__(
        self,
        orchestrator: WeavingOrchestrator,
        scratchpad: Optional[Scratchpad] = None,
        enable_learning: bool = True,
        model_provider: Optional[ModelProvider] = None
    ):
        """
        Initialize advanced refiner.

        Args:
            orchestrator: HoloLoom orchestrator
            scratchpad: Optional scratchpad for tracking
            enable_learning: Whether to learn from refinements
            model_provider: Optional model provider for optimizations
        """
        self.orchestrator = orchestrator
        self.scratchpad = scratchpad
        self.enable_learning = enable_learning
        self.model_provider = model_provider
        self.logger = logging.getLogger(f"{__name__}.AdvancedRefiner")

        # UnifiedMRF for enhanced prompting
        self.mrf = UnifiedMRF()

        # Learning: Track which strategies work best
        self.learned_patterns: Dict[str, RefinementPattern] = {}
        self.strategy_success_rates: Dict[RefinementStrategy, List[float]] = {
            strategy: [] for strategy in RefinementStrategy
        }

    async def refine(
        self,
        query: Query,
        initial_spacetime: Spacetime,
        strategy: Optional[Union[RefinementStrategy, RefinementStrategyType]] = None,
        max_iterations: int = 3,
        quality_threshold: float = 0.9
    ) -> RefinementResult:
        """
        Refine query using specified or auto-selected strategy.

        **UPDATED (Phase 1.4):** Accepts Union[RefinementStrategy, RefinementStrategyType].

        Args:
            query: Original query
            initial_spacetime: Initial result to refine
            strategy: Refinement strategy (auto-selected if None).
                     Accepts RefinementStrategy (deprecated) or RefinementStrategyType.
            max_iterations: Maximum iterations
            quality_threshold: Target quality threshold

        Returns:
            RefinementResult with full trajectory
        """
        # Convert deprecated RefinementStrategy to RefinementStrategyType
        if isinstance(strategy, RefinementStrategy):
            strategy = RefinementStrategy.to_unified_type(strategy)

        # Auto-select strategy if not specified
        if strategy is None:
            strategy = self._select_strategy(query, initial_spacetime)

        self.logger.info(f"Refining with strategy: {strategy.value}")

        # Initialize trajectory tracking
        trajectory: List[QualityMetrics] = [
            self._extract_metrics(initial_spacetime)
        ]

        # Get strategy implementation
        refine_fn = self._get_strategy_function(strategy)

        # Execute refinement iterations
        current_spacetime = initial_spacetime
        for iteration in range(max_iterations):
            # Apply strategy
            refined_spacetime = await refine_fn(
                query=query,
                previous_spacetime=current_spacetime,
                iteration=iteration
            )

            # Track quality
            metrics = self._extract_metrics(refined_spacetime)
            trajectory.append(metrics)

            # Log to scratchpad if available
            if self.scratchpad:
                self.scratchpad.add_entry(
                    thought=f"[Refinement {iteration+1}] Using {strategy.value} strategy",
                    action=f"Quality: {metrics.score():.2f}",
                    observation=f"Confidence: {metrics.confidence:.2f}, Threads: {metrics.threads_activated}",
                    score=metrics.score()
                )

            # Check if threshold reached
            if metrics.score() >= quality_threshold:
                self.logger.info(f"Quality threshold reached after {iteration+1} iterations")
                current_spacetime = refined_spacetime
                break

            # Check for convergence (no improvement)
            if iteration > 0 and metrics.score() <= trajectory[-2].score():
                self.logger.info(f"No improvement detected at iteration {iteration+1}")
                current_spacetime = refined_spacetime
                break

            current_spacetime = refined_spacetime

        # Calculate results
        initial_quality = trajectory[0].score()
        final_quality = trajectory[-1].score()
        improved = final_quality > initial_quality
        improvement_rate = (final_quality - initial_quality) / len(trajectory) if len(trajectory) > 1 else 0.0

        result = RefinementResult(
            final_spacetime=current_spacetime,
            trajectory=trajectory,
            strategy_used=strategy,
            iterations=len(trajectory) - 1,
            improved=improved,
            improvement_rate=improvement_rate
        )

        # Learn from this refinement
        if self.enable_learning:
            self._learn_from_refinement(query, result)

        return result

    def _select_strategy(
        self,
        query: Query,
        spacetime: Spacetime
    ) -> RefinementStrategyType:
        """
        Auto-select best refinement strategy based on query and learned patterns.

        **UPDATED (Phase 1.4):** Returns RefinementStrategyType instead of deprecated RefinementStrategy.

        Args:
            query: Query to refine
            spacetime: Current spacetime

        Returns:
            Selected strategy (RefinementStrategyType)
        """
        # Extract query characteristics
        query_len = len(query.text)
        confidence = spacetime.trace.tool_confidence
        threads_count = len(spacetime.trace.threads_activated)

        # Simple heuristics (can be improved with learning):

        # Low confidence + few threads → REFINE (need more context)
        if confidence < 0.6 and threads_count < 3:
            return RefinementStrategyType.REFINE

        # Medium confidence + many threads → CRITIQUE (refinement needed)
        if 0.6 <= confidence < 0.8 and threads_count >= 3:
            return RefinementStrategyType.CRITIQUE

        # Long, complex query → HOFSTADTER (deep reasoning)
        if query_len > 100:
            return RefinementStrategyType.HOFSTADTER

        # Default: VERIFY (cross-check)
        return RefinementStrategyType.VERIFY

    def _get_strategy_function(
        self,
        strategy: Union[RefinementStrategy, RefinementStrategyType]
    ) -> Callable:
        """
        Get refinement function for strategy.

        **UPDATED (Phase 1.4):** Accepts both RefinementStrategy and RefinementStrategyType.
        """
        # Convert deprecated RefinementStrategy to RefinementStrategyType
        if isinstance(strategy, RefinementStrategy):
            strategy = RefinementStrategy.to_unified_type(strategy)

        strategy_map = {
            RefinementStrategyType.REFINE: self._refine_strategy,
            RefinementStrategyType.CRITIQUE: self._critique_strategy,
            RefinementStrategyType.VERIFY: self._verify_strategy,
            RefinementStrategyType.ELEGANCE: self._elegance_strategy,
            RefinementStrategyType.HOFSTADTER: self._hofstadter_strategy,
        }
        return strategy_map[strategy]

    async def _refine_strategy(
        self,
        query: Query,
        previous_spacetime: Spacetime,
        iteration: int
    ) -> Spacetime:
        """
        REFINE strategy: Iteratively expand query with more context.

        **UPDATED (Phase 1.2):** Uses UnifiedMRF for prompt generation.

        Analyzes why previous result was insufficient and adds context.
        """
        # Use UnifiedMRF to generate refined query
        refined_result = await self.mrf.refine_response(
            query=query.text,
            response=previous_spacetime.response,
            strategy=RefinementStrategyType.REFINE,
            max_iterations=1,  # Single iteration per call
            model=self.model_provider
        )

        # Build expanded query using refined prompt
        expanded_text = refined_result.refined_response
        expanded_query = Query(text=expanded_text, metadata=query.metadata)

        # Re-weave with expanded query
        return await self.orchestrator.weave(expanded_query)

    async def _critique_strategy(
        self,
        query: Query,
        previous_spacetime: Spacetime,
        iteration: int
    ) -> Spacetime:
        """
        CRITIQUE strategy: Identify weaknesses and regenerate.

        **UPDATED (Phase 1.2):** Uses UnifiedMRF for prompt generation.

        Creates a critique of the previous result and uses it to improve.
        """
        # Use UnifiedMRF to generate critique
        refined_result = await self.mrf.refine_response(
            query=query.text,
            response=previous_spacetime.response,
            strategy=RefinementStrategyType.CRITIQUE,
            max_iterations=1,
            model=self.model_provider
        )

        # Build critique query using refined response
        critique_text = refined_result.refined_response
        critique_query = Query(text=critique_text, metadata=query.metadata)

        # Re-weave with critique
        return await self.orchestrator.weave(critique_query)

    async def _verify_strategy(
        self,
        query: Query,
        previous_spacetime: Spacetime,
        iteration: int
    ) -> Spacetime:
        """
        VERIFY strategy: Multi-pass cross-check against multiple sources.

        **UPDATED (Phase 1.2):** Uses UnifiedMRF with iteration-aware prompts.

        Each iteration focuses on a different verification dimension:
        - Pass 1: Accuracy verification
        - Pass 2: Completeness check
        - Pass 3: Consistency validation
        """
        # Use UnifiedMRF to generate verification prompt
        # UnifiedMRF's VERIFY strategy automatically cycles through accuracy→completeness→consistency
        # Note: Multi-pass focus is handled internally by RefinementEngine
        refined_result = await self.mrf.refine_response(
            query=query.text,
            response=previous_spacetime.response,
            strategy=RefinementStrategyType.VERIFY,
            max_iterations=1,
            model=self.model_provider
        )

        # Build verification query using refined response
        verify_text = refined_result.refined_response
        verify_query = Query(text=verify_text, metadata=query.metadata)

        # Re-weave with verification request
        return await self.orchestrator.weave(verify_query)

    async def _elegance_strategy(
        self,
        query: Query,
        previous_spacetime: Spacetime,
        iteration: int
    ) -> Spacetime:
        """
        ELEGANCE strategy: Multi-pass polish for clarity, simplicity, and beauty.

        **UPDATED (Phase 1.2):** Uses UnifiedMRF with iteration-aware prompts.

        Each iteration focuses on a different elegance dimension:
        - Pass 1: Clarity (make it understandable)
        - Pass 2: Simplicity (make it concise)
        - Pass 3: Beauty (make it elegant and well-structured)

        This strategy embraces the philosophy that good answers are:
        - Clear: Easy to understand without ambiguity
        - Simple: No unnecessary complexity
        - Beautiful: Well-organized and aesthetically pleasing
        """
        # Use UnifiedMRF to generate elegance refinement
        # UnifiedMRF's ELEGANCE strategy automatically cycles through clarity→simplicity→beauty
        # Note: Multi-pass focus is handled internally by RefinementEngine
        refined_result = await self.mrf.refine_response(
            query=query.text,
            response=previous_spacetime.response,
            strategy=RefinementStrategyType.ELEGANCE,
            max_iterations=1,
            model=self.model_provider
        )

        # Build elegance query using refined response
        elegance_text = refined_result.refined_response
        elegance_query = Query(text=elegance_text, metadata=query.metadata)

        # Re-weave with elegance refinement
        return await self.orchestrator.weave(elegance_query)

    async def _hofstadter_strategy(
        self,
        query: Query,
        previous_spacetime: Spacetime,
        iteration: int
    ) -> Spacetime:
        """
        HOFSTADTER strategy: Strange loop self-reference.

        **UPDATED (Phase 1.2):** Uses UnifiedMRF for meta-level prompts.

        Uses recursive self-reference to deepen understanding.
        Uses the previous result to inform the next query.
        """
        # Use UnifiedMRF to generate self-referential prompt
        refined_result = await self.mrf.refine_response(
            query=query.text,
            response=previous_spacetime.response,
            strategy=RefinementStrategyType.HOFSTADTER,
            max_iterations=1,
            model=self.model_provider
        )

        # Build self-referential query using refined response
        hofstadter_text = refined_result.refined_response
        hofstadter_query = Query(text=hofstadter_text, metadata=query.metadata)

        # Re-weave with self-reference
        return await self.orchestrator.weave(hofstadter_query)

    def _extract_metrics(self, spacetime: Spacetime) -> QualityMetrics:
        """Extract quality metrics from spacetime"""
        trace = spacetime.trace

        return QualityMetrics(
            confidence=trace.tool_confidence,
            threads_activated=len(trace.threads_activated),
            motifs_detected=len(trace.motifs_detected),
            context_size=len(trace.threads_activated) + len(trace.motifs_detected),
            response_length=len(spacetime.response)
        )

    def _learn_from_refinement(
        self,
        query: Query,
        result: RefinementResult
    ):
        """
        Learn from successful refinement to improve future strategy selection.

        Args:
            query: Query that was refined
            result: Refinement result with trajectory
        """
        if not result.improved:
            return

        # Extract query characteristics
        query_characteristics = {
            "length": len(query.text),
            "initial_confidence": result.trajectory[0].confidence,
            "initial_threads": result.trajectory[0].threads_activated,
        }

        # Create pattern hash
        pattern_hash = f"{result.strategy_used.value}_{query_characteristics['length']//50}"

        # Update or create pattern
        if pattern_hash in self.learned_patterns:
            pattern = self.learned_patterns[pattern_hash]
            improvement = result.trajectory[-1].score() - result.trajectory[0].score()
            pattern.update(improvement)
        else:
            self.learned_patterns[pattern_hash] = RefinementPattern(
                strategy=result.strategy_used,
                initial_quality=result.trajectory[0].score(),
                final_quality=result.trajectory[-1].score(),
                iterations=result.iterations,
                query_characteristics=query_characteristics,
                improvement_rate=result.improvement_rate,
                avg_improvement=result.trajectory[-1].score() - result.trajectory[0].score()
            )

        # Track strategy success
        improvement = result.trajectory[-1].score() - result.trajectory[0].score()
        self.strategy_success_rates[result.strategy_used].append(improvement)

        self.logger.info(
            f"Learned pattern: {result.strategy_used.value} improved "
            f"quality by {improvement:.3f} for query length ~{len(query.text)}"
        )

    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get statistics on strategy performance"""
        stats = {}

        for strategy, improvements in self.strategy_success_rates.items():
            if improvements:
                stats[strategy.value] = {
                    "uses": len(improvements),
                    "avg_improvement": sum(improvements) / len(improvements),
                    "success_rate": sum(1 for x in improvements if x > 0) / len(improvements)
                }
            else:
                stats[strategy.value] = {
                    "uses": 0,
                    "avg_improvement": 0.0,
                    "success_rate": 0.0
                }

        return stats

    def get_learned_patterns(self) -> List[RefinementPattern]:
        """Get all learned refinement patterns"""
        return sorted(
            self.learned_patterns.values(),
            key=lambda p: p.avg_improvement,
            reverse=True
        )


# Convenience function for quick usage
async def refine_with_strategy(
    query: Query,
    initial_spacetime: Spacetime,
    orchestrator: WeavingOrchestrator,
    strategy: Optional[Union[RefinementStrategy, RefinementStrategyType]] = None,
    max_iterations: int = 3,
    quality_threshold: float = 0.9,
    scratchpad: Optional[Scratchpad] = None,
    model_provider: Optional[ModelProvider] = None
) -> RefinementResult:
    """
    Convenience function for one-off advanced refinement.

    **UPDATED (Phase 1.2):** Added model_provider parameter for UnifiedMRF integration.
    **UPDATED (Phase 1.4):** Accepts Union[RefinementStrategy, RefinementStrategyType].

    .. note::
        Prefer using `RefinementStrategyType` from `HoloLoom.prompting.unified_mrf`.
        `RefinementStrategy` is deprecated and will be removed in v2.0.0.

    Args:
        query: Query to refine
        initial_spacetime: Initial result
        orchestrator: HoloLoom orchestrator
        strategy: Refinement strategy (auto-selected if None).
                 Accepts RefinementStrategy (deprecated) or RefinementStrategyType.
        max_iterations: Maximum iterations
        quality_threshold: Target quality
        scratchpad: Optional scratchpad
        model_provider: Optional model provider (Claude/Gemini/GPT/Ollama)

    Returns:
        RefinementResult

    Example:
        # Recommended: Use RefinementStrategyType
        from HoloLoom.prompting.unified_mrf import RefinementStrategyType

        result = await refine_with_strategy(
            query=query,
            initial_spacetime=initial,
            orchestrator=orchestrator,
            strategy=RefinementStrategyType.ELEGANCE,
            model_provider=ModelProvider.CLAUDE
        )

        # Deprecated: RefinementStrategy (emits warning)
        from HoloLoom.recursive.advanced_refinement import RefinementStrategy

        result = await refine_with_strategy(
            query=query,
            initial_spacetime=initial,
            orchestrator=orchestrator,
            strategy=RefinementStrategy.ELEGANCE  # DeprecationWarning
        )
    """
    # Convert deprecated RefinementStrategy to RefinementStrategyType
    if isinstance(strategy, RefinementStrategy):
        strategy = RefinementStrategy.to_unified_type(strategy)

    refiner = AdvancedRefiner(
        orchestrator=orchestrator,
        scratchpad=scratchpad,
        enable_learning=True,
        model_provider=model_provider
    )

    return await refiner.refine(
        query=query,
        initial_spacetime=initial_spacetime,
        strategy=strategy,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold
    )
