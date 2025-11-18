#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5: Multi-Pass Refinement - Refinement Engine

Iteratively improves response quality through multiple refinement passes.

Key Features:
1. Quality-aware refinement (only refine low-confidence responses)
2. Multiple refinement strategies (depth-first, breadth-first, focused)
3. Intelligent stopping criteria (quality threshold, max passes, diminishing returns)
4. Pass-by-pass quality tracking

Usage:
    refiner = RefinementEngine(
        packer=llm_context_packer,
        quality_threshold=0.85,  # Refine if quality < 0.85
        max_passes=3,
        strategy=RefinementStrategy.DEPTH_FIRST
    )

    result = await refiner.refine(
        query="Complex research question",
        awareness_ctx=awareness_context,
        memory_results=memories
    )

    print(f"Quality: {result.initial_quality:.2f} → {result.final_quality:.2f}")
    print(f"Passes: {result.passes_executed}")
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class RefinementStrategy(str, Enum):
    """Refinement strategies for multi-pass improvement"""
    DEPTH_FIRST = "depth_first"      # Deep dive into specific aspects
    BREADTH_FIRST = "breadth_first"  # Expand coverage broadly
    FOCUSED = "focused"              # Focus on weakest quality dimension
    ADAPTIVE = "adaptive"            # Choose strategy based on query


class StoppingCriterion(str, Enum):
    """Criteria for stopping refinement"""
    QUALITY_THRESHOLD = "quality_threshold"  # Quality ≥ threshold
    MAX_PASSES = "max_passes"                # Max passes reached
    DIMINISHING_RETURNS = "diminishing_returns"  # Improvement < threshold
    TIME_LIMIT = "time_limit"                # Total time exceeded


@dataclass
class RefinementPass:
    """Single refinement pass result"""

    pass_number: int
    strategy_used: RefinementStrategy

    # Context modifications
    importance_threshold_adjustment: float  # How much we changed threshold
    compression_disabled: bool  # Did we disable compression for more context?
    budget_multiplier: float  # Budget adjustment (1.0 = no change, 1.5 = +50%)

    # Generation result
    packed_generation: Any  # PackedGeneration from this pass

    # Quality metrics
    quality_before: float
    quality_after: float
    quality_improvement: float

    # Performance
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class RefinementResult:
    """Complete refinement result"""

    # Original query and result
    query: str
    initial_generation: Any  # First PackedGeneration

    # Refinement passes
    passes: List[RefinementPass]

    # Final result
    final_generation: Any  # Best PackedGeneration
    best_pass_number: int  # Which pass produced best result

    # Quality trajectory
    initial_quality: float
    final_quality: float
    total_improvement: float

    # Stopping reason
    stopping_criterion: StoppingCriterion

    # Performance
    passes_executed: int
    total_latency_ms: float
    avg_latency_per_pass_ms: float

    def summary(self) -> str:
        """Human-readable summary"""
        return (
            f"Refinement Summary:\n"
            f"  Query: {self.query[:60]}...\n"
            f"  Quality: {self.initial_quality:.2f} → {self.final_quality:.2f} "
            f"({self.total_improvement:+.2f}, {(self.total_improvement/self.initial_quality)*100:+.1f}%)\n"
            f"  Passes: {self.passes_executed} (best: pass {self.best_pass_number})\n"
            f"  Stopped: {self.stopping_criterion.value}\n"
            f"  Total time: {self.total_latency_ms:.0f}ms "
            f"(avg {self.avg_latency_per_pass_ms:.0f}ms/pass)"
        )


class RefinementEngine:
    """
    Multi-pass refinement engine for iterative quality improvement.

    Refines low-confidence responses through multiple passes with
    intelligent stopping criteria.
    """

    def __init__(
        self,
        packer: Any,  # LLMContextPacker instance
        quality_threshold: float = 0.85,  # Refine if quality < 0.85
        max_passes: int = 3,  # Maximum refinement passes
        min_improvement_per_pass: float = 0.02,  # Minimum 2% improvement to continue
        strategy: RefinementStrategy = RefinementStrategy.ADAPTIVE,
        time_limit_seconds: Optional[float] = None,  # Optional time limit
        enable_compression_disable: bool = True,  # Allow disabling compression for more context
        enable_budget_increase: bool = True  # Allow increasing budget for better quality
    ):
        """
        Initialize refinement engine.

        Args:
            packer: LLMContextPacker instance (Phases 1-4)
            quality_threshold: Minimum quality to accept (refine if below)
            max_passes: Maximum refinement passes
            min_improvement_per_pass: Minimum improvement to justify another pass
            strategy: Refinement strategy
            time_limit_seconds: Optional time limit for total refinement
            enable_compression_disable: Allow disabling compression for more context
            enable_budget_increase: Allow increasing budget
        """
        self.packer = packer
        self.quality_threshold = quality_threshold
        self.max_passes = max_passes
        self.min_improvement = min_improvement_per_pass
        self.strategy = strategy
        self.time_limit = time_limit_seconds
        self.enable_compression_disable = enable_compression_disable
        self.enable_budget_increase = enable_budget_increase

        # Statistics
        self.total_refinements = 0
        self.total_passes_executed = 0
        self.total_quality_improvement = 0.0

    async def refine(
        self,
        query: str,
        awareness_ctx: Any,
        memory_results: Optional[List[Any]] = None,
        max_memories: int = 10,
        force_refinement: bool = False  # Force refinement even if quality > threshold
    ) -> RefinementResult:
        """
        Refine response through multiple passes.

        Args:
            query: User query
            awareness_ctx: Awareness context
            memory_results: Memory retrieval results
            max_memories: Maximum memories to include
            force_refinement: Force refinement even if quality acceptable

        Returns:
            RefinementResult with complete refinement history
        """
        start_time = time.time()

        # 1. Initial generation (Pass 0)
        logger.info(f"Generating initial response for: {query[:50]}...")

        initial_gen = await self.packer.pack_and_generate(
            query=query,
            awareness_context=awareness_ctx,
            memory_results=memory_results,
            max_memories=max_memories
        )

        initial_quality = initial_gen.quality_score.overall

        logger.info(f"Initial quality: {initial_quality:.2f}")

        # 2. Check if refinement needed
        if not force_refinement and initial_quality >= self.quality_threshold:
            logger.info(f"Quality acceptable ({initial_quality:.2f} ≥ {self.quality_threshold:.2f}), skipping refinement")

            return RefinementResult(
                query=query,
                initial_generation=initial_gen,
                passes=[],
                final_generation=initial_gen,
                best_pass_number=0,
                initial_quality=initial_quality,
                final_quality=initial_quality,
                total_improvement=0.0,
                stopping_criterion=StoppingCriterion.QUALITY_THRESHOLD,
                passes_executed=0,
                total_latency_ms=0.0,
                avg_latency_per_pass_ms=0.0
            )

        # 3. Execute refinement passes
        passes = []
        best_generation = initial_gen
        best_quality = initial_quality
        best_pass = 0
        current_quality = initial_quality

        logger.info(f"Starting refinement (quality {initial_quality:.2f} < {self.quality_threshold:.2f})")

        for pass_num in range(1, self.max_passes + 1):
            # Check time limit
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                stopping_criterion = StoppingCriterion.TIME_LIMIT
                logger.info(f"Time limit reached ({self.time_limit}s)")
                break

            # Choose refinement strategy for this pass
            strategy = self._choose_strategy(pass_num, initial_gen, passes)

            # Apply refinement strategy
            pass_start = time.time()

            refined_gen = await self._execute_refinement_pass(
                query=query,
                awareness_ctx=awareness_ctx,
                memory_results=memory_results,
                max_memories=max_memories,
                pass_number=pass_num,
                strategy=strategy,
                previous_generation=passes[-1].packed_generation if passes else initial_gen
            )

            pass_latency = (time.time() - pass_start) * 1000
            refined_quality = refined_gen.quality_score.overall
            improvement = refined_quality - current_quality

            # Record pass
            refinement_pass = RefinementPass(
                pass_number=pass_num,
                strategy_used=strategy,
                importance_threshold_adjustment=0.0,  # TODO: Track actual adjustment
                compression_disabled=False,  # TODO: Track if we disabled compression
                budget_multiplier=1.0,  # TODO: Track budget adjustments
                packed_generation=refined_gen,
                quality_before=current_quality,
                quality_after=refined_quality,
                quality_improvement=improvement,
                latency_ms=pass_latency
            )

            passes.append(refinement_pass)

            logger.info(
                f"Pass {pass_num}: quality {current_quality:.2f} → {refined_quality:.2f} "
                f"({improvement:+.2f}, strategy: {strategy.value})"
            )

            # Update best if improved
            if refined_quality > best_quality:
                best_generation = refined_gen
                best_quality = refined_quality
                best_pass = pass_num

            # Check stopping criteria
            if refined_quality >= self.quality_threshold:
                stopping_criterion = StoppingCriterion.QUALITY_THRESHOLD
                logger.info(f"Quality threshold reached ({refined_quality:.2f} ≥ {self.quality_threshold:.2f})")
                break

            if improvement < self.min_improvement:
                stopping_criterion = StoppingCriterion.DIMINISHING_RETURNS
                logger.info(f"Diminishing returns ({improvement:.3f} < {self.min_improvement:.3f})")
                break

            current_quality = refined_quality

        else:
            # Max passes reached
            stopping_criterion = StoppingCriterion.MAX_PASSES
            logger.info(f"Max passes reached ({self.max_passes})")

        # 4. Build result
        total_latency = (time.time() - start_time) * 1000
        passes_executed = len(passes)

        result = RefinementResult(
            query=query,
            initial_generation=initial_gen,
            passes=passes,
            final_generation=best_generation,
            best_pass_number=best_pass,
            initial_quality=initial_quality,
            final_quality=best_quality,
            total_improvement=best_quality - initial_quality,
            stopping_criterion=stopping_criterion,
            passes_executed=passes_executed,
            total_latency_ms=total_latency,
            avg_latency_per_pass_ms=total_latency / passes_executed if passes_executed > 0 else 0.0
        )

        # Update statistics
        self.total_refinements += 1
        self.total_passes_executed += passes_executed
        self.total_quality_improvement += result.total_improvement

        return result

    def _choose_strategy(
        self,
        pass_number: int,
        initial_generation: Any,
        previous_passes: List[RefinementPass]
    ) -> RefinementStrategy:
        """Choose refinement strategy for this pass"""
        if self.strategy != RefinementStrategy.ADAPTIVE:
            return self.strategy

        # Adaptive strategy selection based on quality dimensions
        quality_score = initial_generation.quality_score

        # Pass 1: Focus on weakest dimension
        if pass_number == 1:
            dimensions = {
                "coherence": quality_score.coherence,
                "completeness": quality_score.completeness,
                "relevance": quality_score.relevance
            }

            weakest = min(dimensions, key=dimensions.get)

            if weakest == "completeness":
                return RefinementStrategy.BREADTH_FIRST  # Expand coverage
            elif weakest == "coherence":
                return RefinementStrategy.FOCUSED  # Focus on structure
            else:
                return RefinementStrategy.DEPTH_FIRST  # Deep dive

        # Pass 2+: Analyze previous improvement patterns
        if previous_passes:
            last_improvement = previous_passes[-1].quality_improvement

            if last_improvement < 0.05:  # Small improvement
                # Try different strategy
                last_strategy = previous_passes[-1].strategy_used

                if last_strategy == RefinementStrategy.DEPTH_FIRST:
                    return RefinementStrategy.BREADTH_FIRST
                else:
                    return RefinementStrategy.DEPTH_FIRST
            else:
                # Continue with successful strategy
                return previous_passes[-1].strategy_used

        return RefinementStrategy.DEPTH_FIRST

    async def _execute_refinement_pass(
        self,
        query: str,
        awareness_ctx: Any,
        memory_results: Optional[List[Any]],
        max_memories: int,
        pass_number: int,
        strategy: RefinementStrategy,
        previous_generation: Any
    ) -> Any:
        """Execute a single refinement pass"""
        # Apply strategy-specific adjustments
        adjustments = self._get_strategy_adjustments(strategy, pass_number)

        # Temporarily modify packer configuration
        original_compression = self.packer.enable_compression
        original_threshold = self.packer.min_importance
        original_budget = getattr(self.packer, 'budget', None)

        try:
            # Apply adjustments
            if adjustments["disable_compression"] and self.enable_compression_disable:
                self.packer.enable_compression = False
                logger.debug(f"Pass {pass_number}: Disabled compression for more context")

            if adjustments["lower_importance_threshold"]:
                self.packer.min_importance *= 0.8  # Lower by 20%
                logger.debug(f"Pass {pass_number}: Lowered importance threshold to {self.packer.min_importance:.2f}")

            if adjustments["increase_budget"] and self.enable_budget_increase:
                if original_budget:
                    self.packer.budget.total = int(original_budget.total * 1.5)
                    logger.debug(f"Pass {pass_number}: Increased budget to {self.packer.budget.total}")

            # Generate refined response
            refined_gen = await self.packer.pack_and_generate(
                query=query,
                awareness_context=awareness_ctx,
                memory_results=memory_results,
                max_memories=max_memories
            )

            return refined_gen

        finally:
            # Restore original configuration
            self.packer.enable_compression = original_compression
            self.packer.min_importance = original_threshold
            if original_budget:
                self.packer.budget = original_budget

    def _get_strategy_adjustments(
        self,
        strategy: RefinementStrategy,
        pass_number: int
    ) -> Dict[str, bool]:
        """Get configuration adjustments for strategy"""
        if strategy == RefinementStrategy.DEPTH_FIRST:
            # Deep dive: more context, no compression
            return {
                "disable_compression": True,
                "lower_importance_threshold": True,
                "increase_budget": True
            }

        elif strategy == RefinementStrategy.BREADTH_FIRST:
            # Broader coverage: more memories, lower threshold
            return {
                "disable_compression": False,
                "lower_importance_threshold": True,
                "increase_budget": True
            }

        elif strategy == RefinementStrategy.FOCUSED:
            # Focused refinement: keep compression, refine specific aspects
            return {
                "disable_compression": False,
                "lower_importance_threshold": False,
                "increase_budget": True
            }

        else:  # ADAPTIVE (shouldn't reach here)
            return {
                "disable_compression": pass_number > 1,  # Disable on later passes
                "lower_importance_threshold": True,
                "increase_budget": True
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get refinement statistics"""
        avg_improvement = (
            self.total_quality_improvement / self.total_refinements
            if self.total_refinements > 0 else 0.0
        )

        avg_passes = (
            self.total_passes_executed / self.total_refinements
            if self.total_refinements > 0 else 0.0
        )

        return {
            "total_refinements": self.total_refinements,
            "total_passes_executed": self.total_passes_executed,
            "avg_passes_per_refinement": avg_passes,
            "avg_quality_improvement": avg_improvement,
            "quality_threshold": self.quality_threshold,
            "max_passes": self.max_passes
        }

    def reset_statistics(self):
        """Reset refinement statistics"""
        self.total_refinements = 0
        self.total_passes_executed = 0
        self.total_quality_improvement = 0.0
