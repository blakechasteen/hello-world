#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Feedback Refiner - Phase 6.1

Extends Phase 5's RefinementEngine to learn from user feedback.

Key Features:
1. Tracks user feedback (thumbs up/down, ratings, corrections)
2. Learns which strategies work for which query types
3. Adapts refinement based on feedback history
4. Provides feedback-based recommendations

Created: 2025-11-18
"""

from typing import Optional, List, Any, Dict
from dataclasses import dataclass

# Import Phase 5 components
from .refinement_engine import (
    RefinementEngine,
    RefinementStrategy,
    RefinementResult,
    StoppingCriterion
)

# Import Phase 6.1 components
from .feedback_tracker import (
    FeedbackTracker,
    FeedbackSignal,
    FeedbackType
)


@dataclass
class FeedbackAwareResult(RefinementResult):
    """
    Extends RefinementResult with feedback tracking capability.

    Adds:
    - feedback_signal: Captured user feedback
    - query_type: Classification for learning
    - recommended_next_strategy: Based on feedback
    """
    feedback_signal: Optional[FeedbackSignal] = None
    query_type: str = "general"
    recommended_next_strategy: Optional[RefinementStrategy] = None


class UserFeedbackRefiner(RefinementEngine):
    """
    Refinement engine that learns from user feedback.

    Extends Phase 5's RefinementEngine with:
    - Feedback tracking and aggregation
    - Strategy selection based on learned preferences
    - Adaptive threshold adjustment from corrections
    - Query type classification for targeted learning

    Usage:
        # Create refiner with feedback learning
        refiner = UserFeedbackRefiner(
            packer=packer,
            quality_threshold=0.85,
            enable_feedback_learning=True
        )

        # Refine query
        result = await refiner.refine(query, awareness_ctx, memories)

        # Track user feedback
        await refiner.track_feedback(
            result=result,
            feedback=FeedbackSignal(
                feedback_type=FeedbackType.THUMBS_UP,
                helpful=True
            )
        )

        # Next refinement uses learned preferences
        result2 = await refiner.refine(similar_query, awareness_ctx, memories)
        # Strategy automatically selected based on feedback history
    """

    def __init__(
        self,
        packer: Any,
        quality_threshold: float = 0.85,
        max_passes: int = 3,
        min_improvement_per_pass: float = 0.02,
        strategy: RefinementStrategy = RefinementStrategy.ADAPTIVE,
        time_limit_seconds: Optional[float] = None,
        enable_compression_disable: bool = True,
        enable_budget_increase: bool = True,
        enable_feedback_learning: bool = True,
        feedback_tracker: Optional[FeedbackTracker] = None
    ):
        """
        Initialize user feedback refiner.

        Args:
            All standard RefinementEngine args, plus:
            enable_feedback_learning: Enable feedback-based learning
            feedback_tracker: Optional external tracker (for shared learning)
        """
        super().__init__(
            packer=packer,
            quality_threshold=quality_threshold,
            max_passes=max_passes,
            min_improvement_per_pass=min_improvement_per_pass,
            strategy=strategy,
            time_limit_seconds=time_limit_seconds,
            enable_compression_disable=enable_compression_disable,
            enable_budget_increase=enable_budget_increase
        )

        self.enable_feedback_learning = enable_feedback_learning
        self.feedback_tracker = feedback_tracker or FeedbackTracker()

        # Track refinements for feedback association
        self.pending_feedback: Dict[str, RefinementResult] = {}

    def classify_query_type(self, query: str, awareness_ctx: Any = None) -> str:
        """
        Classify query type for targeted feedback learning.

        Uses simple heuristics. Can be enhanced with ML classifier.

        Args:
            query: Query text
            awareness_ctx: Optional awareness context

        Returns:
            Query type string (e.g., "technical_explanation")
        """
        query_lower = query.lower()

        # Technical queries
        if any(word in query_lower for word in ["algorithm", "implement", "code", "technical", "function"]):
            if any(word in query_lower for word in ["explain", "what is", "how does"]):
                return "technical_explanation"
            elif any(word in query_lower for word in ["example", "show me"]):
                return "technical_example"
            else:
                return "technical_general"

        # Creative queries
        if any(word in query_lower for word in ["create", "write", "generate", "design"]):
            return "creative"

        # Analytical queries
        if any(word in query_lower for word in ["compare", "analyze", "evaluate", "pros and cons"]):
            return "analytical"

        # Factual queries
        if any(word in query_lower for word in ["what is", "define", "definition"]):
            return "factual_definition"

        # List queries
        if any(word in query_lower for word in ["list", "enumerate", "what are"]):
            return "list"

        # How-to queries
        if query_lower.startswith("how to") or query_lower.startswith("how do"):
            return "howto"

        # Default
        return "general"

    async def refine(
        self,
        query: str,
        awareness_ctx: Any,
        memory_results: Optional[List[Any]] = None,
        max_memories: int = 10,
        force_refinement: bool = False
    ) -> FeedbackAwareResult:
        """
        Refine query with feedback-aware strategy selection.

        If feedback learning enabled, chooses strategy based on:
        1. Feedback history for this query type
        2. Global feedback patterns
        3. Fallback to configured strategy

        Args:
            Standard RefinementEngine.refine args

        Returns:
            FeedbackAwareResult with query type and recommendations
        """
        # Classify query type
        query_type = self.classify_query_type(query, awareness_ctx)

        # Choose strategy based on feedback (if enabled)
        original_strategy = self.strategy

        if self.enable_feedback_learning and self.strategy == RefinementStrategy.ADAPTIVE:
            # Get recommended strategy from feedback
            recommended = self.feedback_tracker.get_recommended_strategy(
                query_type=query_type,
                fallback=original_strategy.value
            )

            # Temporarily set strategy
            try:
                self.strategy = RefinementStrategy(recommended)
            except ValueError:
                # Invalid strategy name, keep original
                pass

        # Execute refinement (calls parent implementation)
        result = await super().refine(
            query=query,
            awareness_ctx=awareness_ctx,
            memory_results=memory_results,
            max_memories=max_memories,
            force_refinement=force_refinement
        )

        # Restore original strategy
        self.strategy = original_strategy

        # Get next recommended strategy based on current feedback
        next_recommended = None
        if self.enable_feedback_learning:
            next_rec = self.feedback_tracker.get_recommended_strategy(
                query_type=query_type,
                fallback="adaptive"
            )
            try:
                next_recommended = RefinementStrategy(next_rec)
            except ValueError:
                pass

        # Create feedback-aware result
        feedback_result = FeedbackAwareResult(
            query=result.query,
            initial_generation=result.initial_generation,
            passes=result.passes,
            final_generation=result.final_generation,
            best_pass_number=result.best_pass_number,
            initial_quality=result.initial_quality,
            final_quality=result.final_quality,
            total_improvement=result.total_improvement,
            stopping_criterion=result.stopping_criterion,
            passes_executed=result.passes_executed,
            total_latency_ms=result.total_latency_ms,
            avg_latency_per_pass_ms=result.avg_latency_per_pass_ms,
            query_type=query_type,
            recommended_next_strategy=next_recommended
        )

        # Store for pending feedback
        self.pending_feedback[query] = feedback_result

        return feedback_result

    async def track_feedback(
        self,
        result: FeedbackAwareResult,
        feedback: FeedbackSignal
    ):
        """
        Track user feedback on a refinement result.

        Learns from feedback to improve future refinements.

        Args:
            result: Refinement result to provide feedback on
            feedback: User feedback signal
        """
        if not self.enable_feedback_learning:
            return

        # Determine which strategy was actually used
        strategy_used = "adaptive"  # Default
        if result.passes:
            # Use strategy from first pass (they may differ in ADAPTIVE mode)
            strategy_used = result.passes[0].strategy_used.value

        # Track feedback
        self.feedback_tracker.track_feedback(
            query=result.query,
            query_type=result.query_type,
            strategy_used=strategy_used,
            feedback=feedback,
            metadata={
                "initial_quality": result.initial_quality,
                "final_quality": result.final_quality,
                "total_improvement": result.total_improvement,
                "passes_executed": result.passes_executed,
                "stopping_criterion": result.stopping_criterion.value
            }
        )

        # Store feedback in result
        result.feedback_signal = feedback

        # Update statistics
        self.total_refinements += 1

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """
        Get feedback statistics.

        Returns:
            Dict with feedback counts, success rates, etc.
        """
        if not self.enable_feedback_learning:
            return {"feedback_learning": "disabled"}

        stats = self.feedback_tracker.get_statistics()

        # Add insights
        stats["insights"] = self.feedback_tracker.get_learning_insights()

        # Add strategy recommendations by query type
        recommendations = {}
        for query_type in self.feedback_tracker.query_type_profiles:
            best_strategy = self.feedback_tracker.get_recommended_strategy(query_type)
            recommendations[query_type] = best_strategy

        stats["strategy_recommendations"] = recommendations

        return stats

    def get_strategy_performance(
        self,
        strategy: str,
        query_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a strategy.

        Args:
            strategy: Strategy name
            query_type: Optional query type filter

        Returns:
            Dict with performance metrics or None
        """
        perf = self.feedback_tracker.get_strategy_performance(strategy, query_type)

        if perf is None:
            return None

        return {
            "strategy": perf.strategy_name,
            "total_uses": perf.total_uses,
            "success_rate": perf.get_success_rate(),
            "average_rating": perf.get_average_rating(),
            "confidence": perf.get_confidence(),
            "overall_score": perf.get_overall_score(),
            "corrections": perf.corrections
        }

    def reset_feedback(self):
        """Reset feedback tracking (for testing)"""
        self.feedback_tracker.reset_statistics()
        self.pending_feedback.clear()


# Convenience function for simple usage
async def refine_with_feedback(
    packer: Any,
    query: str,
    awareness_ctx: Any,
    memory_results: Optional[List[Any]] = None,
    quality_threshold: float = 0.85,
    feedback_tracker: Optional[FeedbackTracker] = None
) -> FeedbackAwareResult:
    """
    Simple wrapper for feedback-aware refinement.

    Args:
        packer: LLMContextPacker instance
        query: Query text
        awareness_ctx: Awareness context
        memory_results: Optional memory results
        quality_threshold: Quality threshold (default: 0.85)
        feedback_tracker: Optional shared feedback tracker

    Returns:
        FeedbackAwareResult

    Example:
        result = await refine_with_feedback(
            packer=packer,
            query="Explain Thompson Sampling",
            awareness_ctx=ctx
        )

        # Track feedback
        await result.refiner.track_feedback(
            result=result,
            feedback=FeedbackSignal(feedback_type=FeedbackType.THUMBS_UP, helpful=True)
        )
    """
    refiner = UserFeedbackRefiner(
        packer=packer,
        quality_threshold=quality_threshold,
        feedback_tracker=feedback_tracker
    )

    result = await refiner.refine(
        query=query,
        awareness_ctx=awareness_ctx,
        memory_results=memory_results
    )

    # Attach refiner to result for convenience
    result.refiner = refiner

    return result
