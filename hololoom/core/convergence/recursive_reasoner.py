#!/usr/bin/env python3
"""
Recursive Reasoner - Core Implementation
========================================

Base implementation of recursive reasoning integrated into HoloLoom's
convergence engine. Migrated from Promptly's recursive_loops.py.

Created: 2025-11-15
Integration: Promptly → HoloLoom convergence layer
"""

import time
from typing import Optional
from datetime import datetime

from hololoom.protocols.types import Query, Features
from hololoom.fabric.spacetime import Spacetime
from hololoom.protocols.recursive_reasoning import (
    RecursiveReasoningProtocol,
    RecursiveConfig,
    ReasoningJournal,
    ReasoningTrace,
    ReasoningStrategy,
    StopCondition,
    QualityScoringProtocol,
    ReasoningResult,
    StopDecision,
    RefinementResult
)


class BaseRecursiveReasoner:
    """
    Base class for recursive reasoning implementations.

    Provides common functionality for all recursive strategies:
    - Iteration loop management
    - Stop condition checking
    - Journal tracking
    - Quality scoring integration
    """

    def __init__(
        self,
        weaving_fn: callable,  # async (Query) -> Spacetime
        quality_scorer: Optional[QualityScoringProtocol] = None
    ):
        """
        Initialize recursive reasoner.

        Args:
            weaving_fn: Function to execute one weaving pass
                        (typically WeavingOrchestrator.weave)
            quality_scorer: Optional quality scoring implementation
                           (defaults to confidence-based scoring)
        """
        self.weaving_fn = weaving_fn
        self.quality_scorer = quality_scorer or DefaultQualityScorer()

    async def reason(
        self,
        query: Query,
        initial_features: Features,
        config: RecursiveConfig
    ) -> ReasoningResult:
        """
        Execute recursive reasoning loop.

        This is the main entry point that orchestrates:
        1. Initial pass
        2. Quality scoring
        3. Iterative refinement (if needed)
        4. Stop condition checking
        5. Journal tracking
        """
        journal = ReasoningJournal(strategy_used=config.strategy)
        start_time = time.time()

        # First pass (use provided initial_features)
        current_spacetime = await self.weaving_fn(query)
        current_quality = await self.quality_scorer.score(current_spacetime, query)

        # Track first iteration
        journal.add_trace(ReasoningTrace(
            iteration=1,
            thought="Initial response generation",
            action=f"weave(query='{query.text[:50]}...')",
            observation=f"Generated response (confidence: {current_quality:.2f})",
            confidence=current_quality,
            features_extracted=initial_features
        ))

        # Refinement loop
        iteration = 2
        while iteration <= config.max_iterations:
            # Check stop conditions
            should_continue, stop_reason = self.should_continue(journal, config)
            if not should_continue:
                break

            # Check time budget
            if config.time_budget_ms:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > config.time_budget_ms:
                    break

            # Execute refinement iteration
            improved_spacetime, trace = await self.refine_iteration(
                current_spacetime,
                journal,
                config
            )

            # Score improved result
            improved_quality = await self.quality_scorer.score(improved_spacetime, query)
            trace.confidence = improved_quality

            # Update journal
            journal.add_trace(trace)

            # Update current best
            if improved_quality > current_quality:
                current_spacetime = improved_spacetime
                current_quality = improved_quality

            iteration += 1

        # Finalize
        journal.final_spacetime = current_spacetime
        return current_spacetime, journal

    def should_continue(
        self,
        journal: ReasoningJournal,
        config: RecursiveConfig
    ) -> StopDecision:
        """Check all stop conditions"""

        # Max iterations reached
        if len(journal.traces) >= config.max_iterations:
            return False, StopCondition.MAX_ITERATIONS

        # Quality threshold met
        latest = journal.get_latest()
        if latest and latest.confidence >= config.quality_threshold:
            return False, StopCondition.QUALITY_THRESHOLD

        # No improvement (converged)
        if journal.converged(config.min_improvement):
            return False, StopCondition.NO_IMPROVEMENT

        # Continue refinement
        return True, None

    async def refine_iteration(
        self,
        previous_spacetime: Spacetime,
        journal: ReasoningJournal,
        config: RecursiveConfig
    ) -> RefinementResult:
        """
        Default refinement: re-weave with context from previous iteration.

        Override in subclasses for strategy-specific refinement.
        """
        # Create refinement query that includes previous output
        refinement_query = Query(
            text=f"""Previous attempt: {previous_spacetime.response}

Please improve this response considering:
- Clarity and completeness
- Accuracy and relevance
- Conciseness

Original query: {previous_spacetime.metadata.get('original_query', '')}
""",
            metadata={"refinement_iteration": len(journal.traces) + 1}
        )

        # Re-weave
        improved = await self.weaving_fn(refinement_query)

        # Create trace
        trace = ReasoningTrace(
            iteration=len(journal.traces) + 1,
            thought="Refining previous response for better quality",
            action=f"weave(refinement_query)",
            observation=f"Generated improved response",
            confidence=0.0,  # Will be scored by caller
            features_extracted=improved.metadata.get('features', None)
        )

        return improved, trace


class DefaultQualityScorer:
    """
    Default quality scoring using spacetime confidence.

    Can be replaced with LLM-based judge for more sophisticated scoring.
    """

    async def score(self, spacetime: Spacetime, query: Query) -> float:
        """
        Score based on spacetime confidence.

        Future enhancements:
        - LLM judge for factual accuracy
        - Embedding similarity to query intent
        - Response completeness heuristics
        """
        # For now, use spacetime confidence (from convergence engine)
        return spacetime.confidence


# Strategy-specific implementations

class RefineReasoner(BaseRecursiveReasoner):
    """
    REFINE strategy: Iterative refinement until quality threshold.

    From Promptly: "Improve output through successive iterations"
    """

    async def refine_iteration(
        self,
        previous_spacetime: Spacetime,
        journal: ReasoningJournal,
        config: RecursiveConfig
    ) -> RefinementResult:
        """Refine by asking for improvements"""

        refinement_query = Query(
            text=f"""Improve this response:

{previous_spacetime.response}

Make it:
1. More accurate and complete
2. Clearer and more concise
3. Better structured

Original query: {previous_spacetime.metadata.get('original_query', '')}
""",
            metadata={"strategy": "refine", "iteration": len(journal.traces) + 1}
        )

        improved = await self.weaving_fn(refinement_query)

        trace = ReasoningTrace(
            iteration=len(journal.traces) + 1,
            thought="Iterative refinement for quality improvement",
            action="refine(previous_response)",
            observation="Generated refined response",
            confidence=0.0,
            features_extracted=improved.metadata.get('features', None)
        )

        return improved, trace


class CritiqueReasoner(BaseRecursiveReasoner):
    """
    CRITIQUE strategy: Generate → Critique → Improve → Repeat.

    From Promptly: "AI critiques its own output and learns from feedback"
    """

    async def refine_iteration(
        self,
        previous_spacetime: Spacetime,
        journal: ReasoningJournal,
        config: RecursiveConfig
    ) -> RefinementResult:
        """First critique, then improve based on critique"""

        # Step 1: Generate critique
        critique_query = Query(
            text=f"""Critique this response:

{previous_spacetime.response}

Identify:
- Errors or inaccuracies
- Missing information
- Unclear explanations
- Areas for improvement

Be specific and constructive.
""",
            metadata={"strategy": "critique", "phase": "critique"}
        )

        critique_spacetime = await self.weaving_fn(critique_query)
        critique = critique_spacetime.response

        # Step 2: Improve based on critique
        improve_query = Query(
            text=f"""Original response:
{previous_spacetime.response}

Critique:
{critique}

Generate an improved response addressing the critique.
""",
            metadata={"strategy": "critique", "phase": "improve"}
        )

        improved = await self.weaving_fn(improve_query)

        trace = ReasoningTrace(
            iteration=len(journal.traces) + 1,
            thought=f"Self-critique identified: {critique[:100]}...",
            action="critique() → improve()",
            observation="Generated improved response based on self-critique",
            confidence=0.0,
            features_extracted=improved.metadata.get('features', None),
            metadata={"critique": critique}
        )

        return improved, trace


class DecomposeReasoner(BaseRecursiveReasoner):
    """
    DECOMPOSE strategy: Break down → Solve → Combine.

    From Promptly: "Split into sub-problems, solve independently, combine"
    """

    async def refine_iteration(
        self,
        previous_spacetime: Spacetime,
        journal: ReasoningJournal,
        config: RecursiveConfig
    ) -> RefinementResult:
        """Decompose problem, solve parts, synthesize"""

        # Step 1: Decompose into sub-problems
        decompose_query = Query(
            text=f"""Break down this query into 3-5 sub-questions:

{previous_spacetime.metadata.get('original_query', '')}

List each sub-question clearly.
""",
            metadata={"strategy": "decompose", "phase": "decompose"}
        )

        decomposition = await self.weaving_fn(decompose_query)
        sub_questions = decomposition.response

        # Step 2: Solve each sub-problem (simplified: solve all together)
        solve_query = Query(
            text=f"""Answer these sub-questions:

{sub_questions}

Be thorough and detailed for each.
""",
            metadata={"strategy": "decompose", "phase": "solve"}
        )

        solutions = await self.weaving_fn(solve_query)

        # Step 3: Synthesize into final answer
        synthesize_query = Query(
            text=f"""Synthesize these answers into a coherent response:

{solutions.response}

Original query: {previous_spacetime.metadata.get('original_query', '')}
""",
            metadata={"strategy": "decompose", "phase": "synthesize"}
        )

        improved = await self.weaving_fn(synthesize_query)

        trace = ReasoningTrace(
            iteration=len(journal.traces) + 1,
            thought="Decomposing problem into sub-questions",
            action="decompose() → solve() → synthesize()",
            observation="Synthesized comprehensive answer from parts",
            confidence=0.0,
            features_extracted=improved.metadata.get('features', None),
            metadata={"sub_questions": sub_questions}
        )

        return improved, trace


class ExploreReasoner(BaseRecursiveReasoner):
    """
    EXPLORE strategy: Multiple approaches → Compare → Synthesize best.

    From Promptly: "Generate N different solutions, compare and rank"
    """

    async def refine_iteration(
        self,
        previous_spacetime: Spacetime,
        journal: ReasoningJournal,
        config: RecursiveConfig
    ) -> RefinementResult:
        """Generate multiple approaches and synthesize best elements"""

        # Generate multiple approaches (parallel would be better)
        approaches = []
        for i in range(config.parallel_explore_branches):
            approach_query = Query(
                text=f"""Approach #{i+1}: Answer this query from a different angle:

{previous_spacetime.metadata.get('original_query', '')}

Be creative and thorough.
""",
                metadata={"strategy": "explore", "branch": i}
            )

            approach = await self.weaving_fn(approach_query)
            approaches.append(approach.response)

        # Synthesize best elements
        synthesize_query = Query(
            text=f"""Compare these {len(approaches)} approaches and synthesize the best answer:

{chr(10).join(f"Approach {i+1}: {a}" for i, a in enumerate(approaches))}

Original query: {previous_spacetime.metadata.get('original_query', '')}
""",
            metadata={"strategy": "explore", "phase": "synthesize"}
        )

        improved = await self.weaving_fn(synthesize_query)

        trace = ReasoningTrace(
            iteration=len(journal.traces) + 1,
            thought=f"Exploring {len(approaches)} different approaches",
            action=f"explore({len(approaches)} branches) → synthesize()",
            observation="Synthesized best elements from multiple approaches",
            confidence=0.0,
            features_extracted=improved.metadata.get('features', None),
            metadata={"approaches": approaches}
        )

        return improved, trace


class HofstadterReasoner(BaseRecursiveReasoner):
    """
    HOFSTADTER strategy: Strange loops (self-reference, meta-reasoning).

    From Promptly: "Meta-cognition, 'What is consciousness?' type questions"

    This is the most experimental strategy, for questions that require
    reasoning about reasoning itself.
    """

    async def refine_iteration(
        self,
        previous_spacetime: Spacetime,
        journal: ReasoningJournal,
        config: RecursiveConfig
    ) -> RefinementResult:
        """Meta-reasoning: reason about the reasoning process itself"""

        # Meta-level query: reason about how to reason about the query
        meta_query = Query(
            text=f"""Consider this question and previous answer:

QUESTION: {previous_spacetime.metadata.get('original_query', '')}

PREVIOUS ANSWER: {previous_spacetime.response}

Now step back and think meta-cognitively:
1. What assumptions are embedded in this question?
2. What does the question reveal about the questioner's mental model?
3. Are we asking the right question, or is there a deeper question?
4. How does the act of answering change the answer itself?

Then provide an improved answer that incorporates this meta-level understanding.
""",
            metadata={"strategy": "hofstadter", "depth": len(journal.traces)}
        )

        improved = await self.weaving_fn(meta_query)

        trace = ReasoningTrace(
            iteration=len(journal.traces) + 1,
            thought="Meta-reasoning: examining assumptions and mental models",
            action="meta_reason() → self_reference()",
            observation="Generated meta-cognitively aware response",
            confidence=0.0,
            features_extracted=improved.metadata.get('features', None),
            metadata={"meta_level": "recursive_self_reference"}
        )

        return improved, trace


# Strategy factory
def create_recursive_reasoner(
    strategy: ReasoningStrategy,
    weaving_fn: callable,
    quality_scorer: Optional[QualityScoringProtocol] = None
) -> BaseRecursiveReasoner:
    """
    Factory to create appropriate reasoner for strategy.

    Args:
        strategy: Which recursive strategy to use
        weaving_fn: Function to execute one weaving pass
        quality_scorer: Optional custom quality scorer

    Returns:
        Configured recursive reasoner
    """
    strategy_map = {
        ReasoningStrategy.REFINE: RefineReasoner,
        ReasoningStrategy.CRITIQUE: CritiqueReasoner,
        ReasoningStrategy.DECOMPOSE: DecomposeReasoner,
        ReasoningStrategy.EXPLORE: ExploreReasoner,
        ReasoningStrategy.HOFSTADTER: HofstadterReasoner,
    }

    reasoner_class = strategy_map.get(strategy, RefineReasoner)
    return reasoner_class(weaving_fn, quality_scorer)
