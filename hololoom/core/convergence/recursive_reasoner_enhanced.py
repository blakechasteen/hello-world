"""
Enhanced Recursive Reasoner - Complete Self-Improving System
==============================================================

Integrates:
- Query decomposition (multi-level breaking)
- Strategy selection (Thompson Sampling learning)
- Convergence detection (multi-criteria)
- Complete provenance tracking

This is the main entry point for recursive reasoning with all Phase 1-5 features.

Created: 2025-11-20
Author: HoloLoom Team
"""

import logging
import time
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

from hololoom.core.protocols.recursive_reasoning import (
    RecursiveConfig,
    RecursiveResult,
    RefinementStep,
    ReasoningJournal,
    ReasoningStrategy,
    StopCondition
)
from hololoom.core.convergence.query_decomposition import QueryDecomposer
from hololoom.core.convergence.refinement_strategies import StrategySelector
from hololoom.core.convergence.detectors import (
    create_multi_criteria_detector,
    ConvergenceDetector
)

logger = logging.getLogger(__name__)


# ============================================================================
# Enhanced Recursive Reasoner
# ============================================================================

class EnhancedRecursiveReasoner:
    """
    Complete recursive reasoner with:
    - Automatic convergence detection
    - Multi-level query decomposition
    - Strategy selection with Thompson Sampling
    - Self-evaluation and improvement tracking
    - Complete provenance logging
    """

    def __init__(
        self,
        department: Any,  # DepartmentProtocol (RAG Department)
        config: Optional[RecursiveConfig] = None,
        decomposer: Optional[QueryDecomposer] = None,
        strategy_selector: Optional[StrategySelector] = None,
        convergence_detector: Optional[ConvergenceDetector] = None
    ):
        """
        Initialize enhanced recursive reasoner.

        Args:
            department: Department to execute queries (RAG Department)
            config: Recursive reasoning configuration
            decomposer: Optional query decomposer
            strategy_selector: Optional strategy selector
            convergence_detector: Optional convergence detector
        """
        self.department = department
        self.config = config or RecursiveConfig()

        # Initialize components
        self.decomposer = decomposer or QueryDecomposer(max_depth=3, max_sub_queries=5)
        self.strategy_selector = strategy_selector or StrategySelector(enable_learning=self.config.enable_learning)
        self.convergence_detector = convergence_detector or create_multi_criteria_detector(
            confidence_threshold=self.config.quality_threshold,
            min_improvement=self.config.min_improvement,
            plateau_window=3,
            max_iterations=self.config.max_iterations,
            logic="OR"
        )

        logger.info(
            f"EnhancedRecursiveReasoner initialized: "
            f"strategy={self.config.strategy.value}, max_iterations={self.config.max_iterations}, "
            f"learning={'on' if self.config.enable_learning else 'off'}"
        )

    async def reason(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> RecursiveResult:
        """
        Execute recursive reasoning with automatic convergence.

        Process:
        1. Check if query needs decomposition
        2. Initial response generation
        3. Quality evaluation
        4. Iterative refinement until convergence
        5. Learning from outcome

        Args:
            query: Query to reason about
            context: Optional context dictionary

        Returns:
            RecursiveResult with final answer and complete trace
        """
        start_time = time.time()
        context = context or {}

        logger.info(f"Starting recursive reasoning: {query[:50]}...")

        # Step 1: Check if decomposition needed
        if self.config.enable_decomposition:
            decomposition = await self.decomposer.decompose(query, complexity_threshold=0.7)

            if len(decomposition.sub_queries) > 1:
                logger.info(f"Query decomposed into {len(decomposition.sub_queries)} sub-queries")
                return await self._reason_decomposed(query, decomposition, context, start_time)

        # Step 2: Direct reasoning (no decomposition)
        return await self._reason_direct(query, context, start_time)

    async def _reason_direct(
        self,
        query: str,
        context: Dict,
        start_time: float
    ) -> RecursiveResult:
        """
        Direct recursive reasoning (no decomposition).

        Args:
            query: Query to reason about
            context: Context dictionary
            start_time: Start time (for latency tracking)

        Returns:
            RecursiveResult
        """
        refinement_steps: List[RefinementStep] = []

        # Initial response
        logger.info("Generating initial response...")
        initial_result = await self.department.execute({
            "query": query,
            "mode": "verify",
            "max_sources": 5
        })

        initial_confidence = initial_result.confidence.score
        refinement_steps.append(RefinementStep(
            iteration=1,
            query=query,
            response=initial_result.response.get("answer", ""),
            confidence=initial_confidence,
            improvement=0.0,
            strategy_used="initial",
            reasoning="Initial response generation",
            timestamp=datetime.utcnow()
        ))

        logger.info(f"Initial response: confidence={initial_confidence:.2f}")

        # Check if refinement needed
        if initial_confidence >= self.config.refinement_threshold:
            logger.info(f"Confidence {initial_confidence:.2f} >= threshold {self.config.refinement_threshold}, no refinement needed")
            return self._build_result(query, refinement_steps, start_time, StopCondition.QUALITY_THRESHOLD)

        # Refinement loop
        current_result = initial_result
        iteration = 2

        while iteration <= self.config.max_refinement_iterations:
            # Check convergence
            converged, reason = self.convergence_detector.detect(refinement_steps)
            if converged:
                logger.info(f"Converged: {reason}")
                return self._build_result(query, refinement_steps, start_time, StopCondition.QUALITY_THRESHOLD, convergence_reason=reason)

            # Check time budget
            if self.config.time_budget_ms:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > self.config.time_budget_ms:
                    logger.info(f"Time budget {self.config.time_budget_ms}ms exceeded")
                    return self._build_result(query, refinement_steps, start_time, StopCondition.TIME_BUDGET)

            # Select refinement strategy
            strategy = await self.strategy_selector.select_strategy(
                query=query,
                current_confidence=current_result.confidence.score,
                refinement_history=refinement_steps
            )

            logger.info(f"Iteration {iteration}: Refining with strategy={strategy.value}")

            # Execute refinement
            refined_result, improvement = await self._execute_refinement(
                query=query,
                previous_result=current_result,
                strategy=strategy
            )

            # Track refinement step
            refinement_steps.append(RefinementStep(
                iteration=iteration,
                query=query,
                response=refined_result.response.get("answer", ""),
                confidence=refined_result.confidence.score,
                improvement=improvement,
                strategy_used=strategy.value,
                reasoning=f"Refinement using {strategy.value}",
                timestamp=datetime.utcnow()
            ))

            logger.info(f"  Result: confidence={refined_result.confidence.score:.2f}, improvement={improvement:+.3f}")

            # Learn from outcome
            if self.config.enable_learning:
                query_type = self.strategy_selector.classifier.classify(query)
                await self.strategy_selector.learn_from_outcome(strategy, improvement, query_type)

            # Update current result
            current_result = refined_result
            iteration += 1

        # Max iterations reached
        logger.info(f"Max refinement iterations {self.config.max_refinement_iterations} reached")
        return self._build_result(query, refinement_steps, start_time, StopCondition.MAX_ITERATIONS)

    async def _reason_decomposed(
        self,
        root_query: str,
        decomposition: Any,
        context: Dict,
        start_time: float
    ) -> RecursiveResult:
        """
        Recursive reasoning with query decomposition.

        Args:
            root_query: Original root query
            decomposition: DecompositionTree
            context: Context dictionary
            start_time: Start time

        Returns:
            RecursiveResult
        """
        logger.info(f"Reasoning with decomposition: {len(decomposition.sub_queries)} sub-queries")

        refinement_steps: List[RefinementStep] = []
        sub_results = []

        # Solve each sub-query
        for i, sub_query in enumerate(decomposition.sub_queries):
            logger.info(f"  Sub-query {i+1}/{len(decomposition.sub_queries)}: {sub_query[:50]}...")

            sub_result = await self.department.execute({
                "query": sub_query,
                "mode": "direct",  # Simpler mode for sub-queries
                "max_sources": 3
            })

            sub_results.append(sub_result)

            # Track as refinement step
            refinement_steps.append(RefinementStep(
                iteration=i+1,
                query=sub_query,
                response=sub_result.response.get("answer", ""),
                confidence=sub_result.confidence.score,
                improvement=0.0,  # N/A for decomposed queries
                strategy_used=f"sub_query_{i+1}",
                reasoning=f"Sub-query {i+1}/{len(decomposition.sub_queries)}",
                timestamp=datetime.utcnow()
            ))

        # Synthesize sub-answers
        logger.info("Synthesizing sub-answers...")
        synthesized_answer = await self.decomposer.synthesize_sub_answers(
            sub_results,
            synthesis_strategy=decomposition.synthesis_strategy
        )

        # Final step: synthesized answer
        avg_confidence = sum(r.confidence.score for r in sub_results) / len(sub_results)
        refinement_steps.append(RefinementStep(
            iteration=len(refinement_steps) + 1,
            query=root_query,
            response=synthesized_answer,
            confidence=avg_confidence,
            improvement=0.0,
            strategy_used="synthesis",
            reasoning=f"Synthesized {len(decomposition.sub_queries)} sub-answers",
            timestamp=datetime.utcnow()
        ))

        logger.info(f"Decomposition complete: avg_confidence={avg_confidence:.2f}")

        return self._build_result(
            root_query,
            refinement_steps,
            start_time,
            StopCondition.QUALITY_THRESHOLD,
            convergence_reason="Decomposition synthesis complete",
            decomposition=decomposition
        )

    async def _execute_refinement(
        self,
        query: str,
        previous_result: Any,
        strategy: Any
    ) -> tuple:
        """
        Execute refinement with selected strategy.

        Args:
            query: Original query
            previous_result: Previous department response
            strategy: RefinementStrategy

        Returns:
            (refined_result, improvement) tuple
        """
        from hololoom.core.convergence.refinement_strategies import StrategyExecutor

        # Get execution parameters for strategy
        params = StrategyExecutor.get_execution_params(strategy)

        # Execute refinement
        refined_result = await self.department.execute({
            "query": query,
            **params
        })

        # Calculate improvement
        improvement = refined_result.confidence.score - previous_result.confidence.score

        return refined_result, improvement

    def _build_result(
        self,
        query: str,
        refinement_steps: List[RefinementStep],
        start_time: float,
        stop_condition: StopCondition,
        convergence_reason: str = "",
        decomposition: Optional[Any] = None
    ) -> RecursiveResult:
        """
        Build RecursiveResult from refinement history.

        Args:
            query: Original query
            refinement_steps: List of refinement steps
            start_time: Start timestamp
            stop_condition: Why reasoning stopped
            convergence_reason: Detailed convergence reason
            decomposition: Optional DecompositionTree

        Returns:
            RecursiveResult
        """
        total_latency_ms = (time.time() - start_time) * 1000
        final_step = refinement_steps[-1]

        improvement_trajectory = [step.confidence for step in refinement_steps]

        # Build journal (for compatibility)
        journal = ReasoningJournal(strategy_used=self.config.strategy)
        # Note: Would need to convert RefinementStep → ReasoningTrace if needed

        result = RecursiveResult(
            final_response=final_step.response,
            final_confidence=final_step.confidence,
            iterations=len(refinement_steps),
            refinement_steps=refinement_steps,
            convergence_reason=convergence_reason or stop_condition.value,
            total_latency_ms=total_latency_ms,
            improvement_trajectory=improvement_trajectory,
            journal=journal,
            decomposition=decomposition,
            metadata={
                "query": query,
                "stop_condition": stop_condition.value,
                "initial_confidence": refinement_steps[0].confidence,
                "final_confidence": final_step.confidence,
                "total_improvement": final_step.confidence - refinement_steps[0].confidence,
            }
        )

        logger.info(
            f"Recursive reasoning complete: "
            f"iterations={result.iterations}, "
            f"confidence={result.final_confidence:.2f}, "
            f"latency={result.total_latency_ms:.1f}ms"
        )

        return result


# ============================================================================
# Factory
# ============================================================================

def create_recursive_reasoner(
    department: Any,
    config: Optional[RecursiveConfig] = None,
    enable_decomposition: bool = True,
    enable_learning: bool = True
) -> EnhancedRecursiveReasoner:
    """
    Create enhanced recursive reasoner.

    Args:
        department: Department to execute queries
        config: Optional configuration
        enable_decomposition: Enable query decomposition
        enable_learning: Enable strategy learning

    Returns:
        Configured EnhancedRecursiveReasoner
    """
    if config is None:
        config = RecursiveConfig(
            enable_decomposition=enable_decomposition,
            enable_learning=enable_learning
        )

    return EnhancedRecursiveReasoner(
        department=department,
        config=config
    )
