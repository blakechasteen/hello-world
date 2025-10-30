"""
Advanced Recursive Refinement - Phase 4
========================================
Multiple refinement strategies, quality trajectory tracking, and learning from
successful refinements.

Refinement Strategies:
- REFINE: Iteratively expand and improve query
- CRITIQUE: Self-critique result and regenerate
- VERIFY: Multi-pass cross-check against multiple sources
- ELEGANCE: Iteratively polish for clarity, simplicity, and beauty
- HOFSTADTER: Strange loop self-reference for deep reasoning

This module enhances Phase 1's basic refinement with sophisticated strategies
that learn from successful refinement patterns.

Multi-Pass Philosophy:
The ELEGANCE and VERIFY strategies embrace multiple passes:
- Each pass improves a specific dimension (clarity, accuracy, simplicity)
- Quality trajectory shows incremental improvement
- Learning identifies when additional passes yield diminishing returns
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime

from HoloLoom.documentation.types import Query, Spacetime
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from Promptly.promptly.recursive_loops import Scratchpad, ScratchpadEntry


class RefinementStrategy(Enum):
    """Available refinement strategies"""
    REFINE = "refine"          # Iterative expansion and improvement
    CRITIQUE = "critique"      # Self-critique and regenerate
    VERIFY = "verify"          # Multi-pass cross-check multiple sources
    ELEGANCE = "elegance"      # Multi-pass polish for clarity and simplicity
    HOFSTADTER = "hofstadter"  # Strange loop self-reference


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
    """Result of refinement process with full trajectory"""
    final_spacetime: Spacetime
    trajectory: List[QualityMetrics]
    strategy_used: RefinementStrategy
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

    This refiner can:
    1. Apply different refinement strategies based on query type
    2. Track quality trajectories across iterations
    3. Learn which strategies work best for which queries
    4. Adapt strategy selection based on past successes
    """

    def __init__(
        self,
        orchestrator: WeavingOrchestrator,
        scratchpad: Optional[Scratchpad] = None,
        enable_learning: bool = True
    ):
        """
        Initialize advanced refiner.

        Args:
            orchestrator: HoloLoom orchestrator
            scratchpad: Optional scratchpad for tracking
            enable_learning: Whether to learn from refinements
        """
        self.orchestrator = orchestrator
        self.scratchpad = scratchpad
        self.enable_learning = enable_learning
        self.logger = logging.getLogger(f"{__name__}.AdvancedRefiner")

        # Learning: Track which strategies work best
        self.learned_patterns: Dict[str, RefinementPattern] = {}
        self.strategy_success_rates: Dict[RefinementStrategy, List[float]] = {
            strategy: [] for strategy in RefinementStrategy
        }

    async def refine(
        self,
        query: Query,
        initial_spacetime: Spacetime,
        strategy: Optional[RefinementStrategy] = None,
        max_iterations: int = 3,
        quality_threshold: float = 0.9
    ) -> RefinementResult:
        """
        Refine query using specified or auto-selected strategy.

        Args:
            query: Original query
            initial_spacetime: Initial result to refine
            strategy: Refinement strategy (auto-selected if None)
            max_iterations: Maximum iterations
            quality_threshold: Target quality threshold

        Returns:
            RefinementResult with full trajectory
        """
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
    ) -> RefinementStrategy:
        """
        Auto-select best refinement strategy based on query and learned patterns.

        Args:
            query: Query to refine
            spacetime: Current spacetime

        Returns:
            Selected strategy
        """
        # Extract query characteristics
        query_len = len(query.text)
        confidence = spacetime.trace.tool_confidence
        threads_count = len(spacetime.trace.threads_activated)

        # Simple heuristics (can be improved with learning):

        # Low confidence + few threads → REFINE (need more context)
        if confidence < 0.6 and threads_count < 3:
            return RefinementStrategy.REFINE

        # Medium confidence + many threads → CRITIQUE (refinement needed)
        if 0.6 <= confidence < 0.8 and threads_count >= 3:
            return RefinementStrategy.CRITIQUE

        # Long, complex query → HOFSTADTER (deep reasoning)
        if query_len > 100:
            return RefinementStrategy.HOFSTADTER

        # Default: VERIFY (cross-check)
        return RefinementStrategy.VERIFY

    def _get_strategy_function(
        self,
        strategy: RefinementStrategy
    ) -> Callable:
        """Get refinement function for strategy"""
        strategy_map = {
            RefinementStrategy.REFINE: self._refine_strategy,
            RefinementStrategy.CRITIQUE: self._critique_strategy,
            RefinementStrategy.VERIFY: self._verify_strategy,
            RefinementStrategy.ELEGANCE: self._elegance_strategy,
            RefinementStrategy.HOFSTADTER: self._hofstadter_strategy,
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

        Analyzes why previous result was insufficient and adds context.
        """
        # Analyze what's missing
        trace = previous_spacetime.trace
        threads = trace.threads_activated
        motifs = trace.motifs_detected

        # Build expanded query
        expansion_parts = [query.text]

        if len(threads) < 3:
            expansion_parts.append("Please provide more context and background information.")

        if len(motifs) < 2:
            expansion_parts.append("Clarify the key concepts and their relationships.")

        if trace.tool_confidence < 0.7:
            expansion_parts.append("Include specific examples and details.")

        expanded_text = " ".join(expansion_parts)
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

        Creates a critique of the previous result and uses it to improve.
        """
        # Create critique query
        critique_text = (
            f"{query.text}\n\n"
            f"Previous attempt had confidence {previous_spacetime.trace.tool_confidence:.2f}. "
            f"Improve by addressing: completeness, accuracy, and clarity."
        )

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

        Each iteration focuses on a different verification dimension:
        - Pass 1: Accuracy verification
        - Pass 2: Completeness check
        - Pass 3: Consistency validation
        """
        # Multi-pass verification focuses
        verification_focuses = [
            "Verify the accuracy of all factual claims",
            "Check for completeness - are there gaps or missing information?",
            "Validate internal consistency - do all parts align?"
        ]

        focus = verification_focuses[iteration % len(verification_focuses)]

        # Create verification query with iteration-specific focus
        verify_text = (
            f"{query.text}\n\n"
            f"Verification Pass {iteration + 1}: {focus}\n"
            f"Cross-check this information across multiple sources."
        )

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

        Each iteration focuses on a different elegance dimension:
        - Pass 1: Clarity (make it understandable)
        - Pass 2: Simplicity (make it concise)
        - Pass 3: Beauty (make it elegant and well-structured)

        This strategy embraces the philosophy that good answers are:
        - Clear: Easy to understand without ambiguity
        - Simple: No unnecessary complexity
        - Beautiful: Well-organized and aesthetically pleasing
        """
        # Multi-pass elegance focuses
        elegance_focuses = [
            {
                "dimension": "Clarity",
                "instruction": "Improve clarity - make the explanation crystal clear and unambiguous. "
                              "Remove jargon, add concrete examples, clarify any confusing parts."
            },
            {
                "dimension": "Simplicity",
                "instruction": "Improve simplicity - make it more concise without losing meaning. "
                              "Remove redundancy, streamline structure, use simpler language where possible."
            },
            {
                "dimension": "Beauty",
                "instruction": "Improve elegance - organize for maximum aesthetic and conceptual beauty. "
                              "Create logical flow, use parallel structure, achieve balance and harmony."
            }
        ]

        focus = elegance_focuses[iteration % len(elegance_focuses)]

        # Create elegance refinement query
        elegance_text = (
            f"{query.text}\n\n"
            f"Elegance Pass {iteration + 1} - {focus['dimension']}:\n"
            f"{focus['instruction']}\n\n"
            f"Previous response:\n{previous_spacetime.response[:300]}...\n\n"
            f"Refine this to be more {focus['dimension'].lower()}."
        )

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

        Uses recursive self-reference to deepen understanding.
        Uses the previous result to inform the next query.
        """
        # Create self-referential query
        hofstadter_text = (
            f"{query.text}\n\n"
            f"Building on the previous understanding: {previous_spacetime.response[:200]}...\n"
            f"Now deepen this by exploring the meta-level: What patterns or principles "
            f"underlie this? How does this connect to broader concepts?"
        )

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
    strategy: Optional[RefinementStrategy] = None,
    max_iterations: int = 3,
    quality_threshold: float = 0.9,
    scratchpad: Optional[Scratchpad] = None
) -> RefinementResult:
    """
    Convenience function for one-off advanced refinement.

    Args:
        query: Query to refine
        initial_spacetime: Initial result
        orchestrator: HoloLoom orchestrator
        strategy: Refinement strategy (auto-selected if None)
        max_iterations: Maximum iterations
        quality_threshold: Target quality
        scratchpad: Optional scratchpad

    Returns:
        RefinementResult
    """
    refiner = AdvancedRefiner(
        orchestrator=orchestrator,
        scratchpad=scratchpad,
        enable_learning=True
    )

    return await refiner.refine(
        query=query,
        initial_spacetime=initial_spacetime,
        strategy=strategy,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold
    )
