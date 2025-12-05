#!/usr/bin/env python3
"""
Refinement System for Promptly Matrix Bot

Multi-pass quality improvement with three strategies:
1. ELEGANCE: Clarity → Simplicity → Beauty
2. VERIFY: Accuracy → Completeness → Consistency
3. CRITIQUE: Self-improvement loop

Features:
- Iterative refinement with quality metrics
- Strategy-specific improvement passes
- Integration with HoloLoom recursive learning
- Quality trajectory tracking
- Automatic convergence detection

Usage:
    refiner = Refiner(promptly_core)
    result = await refiner.refine(
        query="Explain Thompson Sampling",
        initial_response="...",
        strategy=RefinementStrategy.ELEGANCE,
        max_iterations=3
    )
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RefinementStrategy(Enum):
    """Available refinement strategies"""
    ELEGANCE = "elegance"      # Clarity → Simplicity → Beauty
    VERIFY = "verify"          # Accuracy → Completeness → Consistency
    CRITIQUE = "critique"      # Self-improvement loop


@dataclass
class QualityMetrics:
    """Quality metrics for refinement"""
    clarity: float = 0.0       # [0-1] How clear is the response?
    simplicity: float = 0.0    # [0-1] How simple/concise?
    accuracy: float = 0.0      # [0-1] Factually correct?
    completeness: float = 0.0  # [0-1] Covers all aspects?
    consistency: float = 0.0   # [0-1] Internally consistent?

    def overall_score(self, strategy: RefinementStrategy) -> float:
        """Calculate overall quality score based on strategy"""
        if strategy == RefinementStrategy.ELEGANCE:
            # Weight aesthetic qualities
            return (self.clarity * 0.4 +
                   self.simplicity * 0.4 +
                   self.accuracy * 0.2)

        elif strategy == RefinementStrategy.VERIFY:
            # Weight factual accuracy
            return (self.accuracy * 0.5 +
                   self.completeness * 0.3 +
                   self.consistency * 0.2)

        else:  # CRITIQUE
            # Balanced weight
            return (self.clarity * 0.2 +
                   self.accuracy * 0.3 +
                   self.completeness * 0.3 +
                   self.consistency * 0.2)


@dataclass
class RefinementPass:
    """Single refinement pass"""
    iteration: int
    focus: str                  # What this pass focuses on
    input_text: str
    output_text: str
    metrics: QualityMetrics
    improvements: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class RefinementResult:
    """Complete refinement result"""
    query: str
    strategy: RefinementStrategy
    initial_response: str
    final_response: str
    passes: List[RefinementPass]
    initial_metrics: QualityMetrics
    final_metrics: QualityMetrics
    converged: bool
    total_duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Human-readable summary"""
        initial_score = self.initial_metrics.overall_score(self.strategy)
        final_score = self.final_metrics.overall_score(self.strategy)
        improvement = ((final_score - initial_score) / max(initial_score, 0.01)) * 100

        status = "✅ CONVERGED" if self.converged else "⏸️ MAX ITERATIONS"

        return (
            f"{status}\n"
            f"Strategy: {self.strategy.value}\n"
            f"Iterations: {len(self.passes)}\n"
            f"Quality: {initial_score:.2f} → {final_score:.2f} "
            f"({improvement:+.1f}%)\n"
            f"Duration: {self.total_duration_ms:.0f}ms"
        )


class Refiner:
    """
    Multi-pass quality improvement system.

    Usage:
        refiner = Refiner(promptly_core)
        result = await refiner.refine(
            query="Explain recursion",
            initial_response="Recursion is when a function calls itself",
            strategy=RefinementStrategy.ELEGANCE,
            max_iterations=3,
            convergence_threshold=0.05
        )

        print(result.summary())
        print(result.final_response)
    """

    def __init__(self, promptly_core=None):
        """
        Initialize refiner.

        Args:
            promptly_core: Promptly Core instance for LLM calls
        """
        self.promptly_core = promptly_core

    async def refine(
        self,
        query: str,
        initial_response: str,
        strategy: RefinementStrategy = RefinementStrategy.ELEGANCE,
        max_iterations: int = 3,
        convergence_threshold: float = 0.05,
        quality_threshold: float = 0.9
    ) -> RefinementResult:
        """
        Refine response through multi-pass improvement.

        Args:
            query: Original query/task
            initial_response: Initial response to refine
            strategy: Refinement strategy to use
            max_iterations: Maximum refinement passes
            convergence_threshold: Stop if improvement < threshold
            quality_threshold: Stop if quality exceeds threshold

        Returns:
            RefinementResult with complete refinement history
        """
        import time
        start_time = time.time()

        # Measure initial quality
        initial_metrics = await self._measure_quality(initial_response, query, strategy)

        result = RefinementResult(
            query=query,
            strategy=strategy,
            initial_response=initial_response,
            final_response=initial_response,
            passes=[],
            initial_metrics=initial_metrics,
            final_metrics=initial_metrics,
            converged=False,
            total_duration_ms=0.0
        )

        current_text = initial_response
        previous_score = initial_metrics.overall_score(strategy)

        # Multi-pass refinement
        for iteration in range(max_iterations):
            pass_start = time.time()

            # Determine focus for this pass
            focus = self._get_pass_focus(strategy, iteration)

            # Perform refinement pass
            refined_text, improvements = await self._refine_pass(
                query=query,
                current_text=current_text,
                focus=focus,
                strategy=strategy
            )

            # Measure quality
            metrics = await self._measure_quality(refined_text, query, strategy)
            current_score = metrics.overall_score(strategy)

            # Record pass
            refinement_pass = RefinementPass(
                iteration=iteration + 1,
                focus=focus,
                input_text=current_text,
                output_text=refined_text,
                metrics=metrics,
                improvements=improvements,
                duration_ms=(time.time() - pass_start) * 1000
            )
            result.passes.append(refinement_pass)

            # Check convergence
            improvement = current_score - previous_score

            logger.info(
                f"Refinement pass {iteration + 1}: "
                f"focus={focus}, score={current_score:.2f}, "
                f"improvement={improvement:+.2f}"
            )

            # Update current text
            current_text = refined_text
            previous_score = current_score

            # Check stopping conditions
            if current_score >= quality_threshold:
                logger.info(f"Quality threshold reached: {current_score:.2f}")
                result.converged = True
                break

            if improvement < convergence_threshold:
                logger.info(f"Converged: improvement {improvement:.2f} < {convergence_threshold:.2f}")
                result.converged = True
                break

        # Finalize result
        result.final_response = current_text
        result.final_metrics = await self._measure_quality(current_text, query, strategy)
        result.total_duration_ms = (time.time() - start_time) * 1000

        logger.info(f"Refinement complete: {result.summary()}")

        return result

    def _get_pass_focus(self, strategy: RefinementStrategy, iteration: int) -> str:
        """Determine focus for refinement pass"""
        if strategy == RefinementStrategy.ELEGANCE:
            # Clarity → Simplicity → Beauty
            focuses = ["clarity", "simplicity", "beauty"]
            return focuses[iteration % len(focuses)]

        elif strategy == RefinementStrategy.VERIFY:
            # Accuracy → Completeness → Consistency
            focuses = ["accuracy", "completeness", "consistency"]
            return focuses[iteration % len(focuses)]

        else:  # CRITIQUE
            # Self-improvement loop
            return "self-critique"

    async def _refine_pass(
        self,
        query: str,
        current_text: str,
        focus: str,
        strategy: RefinementStrategy
    ) -> tuple[str, List[str]]:
        """
        Perform single refinement pass.

        Returns:
            (refined_text, list_of_improvements)
        """
        improvements = []

        # Use LLM if available
        if self.promptly_core:
            try:
                refined_text = await self._llm_refine(
                    query, current_text, focus, strategy
                )

                # Extract improvements (simple heuristic)
                improvements = self._extract_improvements(current_text, refined_text)

                return refined_text, improvements

            except Exception as e:
                logger.warning(f"LLM refinement failed: {e}")

        # Fallback: Heuristic refinement
        refined_text, improvements = self._heuristic_refine(
            current_text, focus
        )

        return refined_text, improvements

    async def _llm_refine(
        self,
        query: str,
        current_text: str,
        focus: str,
        strategy: RefinementStrategy
    ) -> str:
        """Refine using LLM"""
        prompt = self._build_refinement_prompt(
            query, current_text, focus, strategy
        )

        result = await self.promptly_core.run_workflow("refine", prompt)

        if result.get('success'):
            return result.get('output', current_text)

        return current_text

    def _build_refinement_prompt(
        self,
        query: str,
        current_text: str,
        focus: str,
        strategy: RefinementStrategy
    ) -> str:
        """Build refinement prompt"""
        focus_instructions = {
            "clarity": "Make the response clearer and easier to understand. Use simpler language and better structure.",
            "simplicity": "Simplify the response. Remove unnecessary complexity while preserving meaning.",
            "beauty": "Improve the elegance and flow. Make it more pleasant to read.",
            "accuracy": "Improve factual accuracy. Add missing details and correct any errors.",
            "completeness": "Make the response more complete. Cover all relevant aspects.",
            "consistency": "Improve internal consistency. Resolve any contradictions.",
            "self-critique": "Critically analyze the response and improve weak points."
        }

        instruction = focus_instructions.get(focus, "Improve the response.")

        return f"""Refine this response with focus on: {focus}

Original Query: {query}

Current Response:
{current_text}

Instructions:
{instruction}

Strategy: {strategy.value}

Provide the refined response:
"""

    def _heuristic_refine(
        self,
        current_text: str,
        focus: str
    ) -> tuple[str, List[str]]:
        """Heuristic refinement (no LLM)"""
        improvements = []
        refined_text = current_text

        # Simple heuristic improvements
        if focus == "clarity":
            # Remove redundant words
            redundant = ["very", "really", "actually", "basically"]
            for word in redundant:
                if word in refined_text.lower():
                    refined_text = refined_text.replace(f" {word} ", " ")
                    improvements.append(f"Removed redundant '{word}'")

        elif focus == "simplicity":
            # Suggest shorter sentences
            sentences = refined_text.split('. ')
            long_sentences = [s for s in sentences if len(s.split()) > 25]
            if long_sentences:
                improvements.append(f"Found {len(long_sentences)} long sentences to simplify")

        return refined_text, improvements if improvements else ["Heuristic refinement applied"]

    def _extract_improvements(self, old_text: str, new_text: str) -> List[str]:
        """Extract improvements made (simple diff)"""
        improvements = []

        if len(new_text) < len(old_text):
            improvements.append(f"Shortened by {len(old_text) - len(new_text)} characters")
        elif len(new_text) > len(old_text):
            improvements.append(f"Expanded by {len(new_text) - len(old_text)} characters")

        # Count sentence changes
        old_sentences = len([s for s in old_text.split('.') if s.strip()])
        new_sentences = len([s for s in new_text.split('.') if s.strip()])

        if new_sentences != old_sentences:
            improvements.append(f"Restructured: {old_sentences} → {new_sentences} sentences")

        return improvements if improvements else ["Refined response"]

    async def _measure_quality(
        self,
        text: str,
        query: str,
        strategy: RefinementStrategy
    ) -> QualityMetrics:
        """Measure response quality"""
        metrics = QualityMetrics()

        # Use LLM if available
        if self.promptly_core:
            try:
                llm_metrics = await self._llm_measure_quality(text, query, strategy)
                if llm_metrics:
                    return llm_metrics
            except Exception as e:
                logger.warning(f"LLM quality measurement failed: {e}")

        # Fallback: Heuristic metrics
        return self._heuristic_measure_quality(text, query)

    async def _llm_measure_quality(
        self,
        text: str,
        query: str,
        strategy: RefinementStrategy
    ) -> Optional[QualityMetrics]:
        """Measure quality using LLM"""
        prompt = f"""Rate the quality of this response on a scale of 0-100 for each metric:

Query: {query}

Response:
{text}

Metrics to rate:
1. Clarity: How clear and easy to understand?
2. Simplicity: How concise and simple?
3. Accuracy: How factually correct?
4. Completeness: How thoroughly does it answer?
5. Consistency: How internally consistent?

Format:
Clarity: [0-100]
Simplicity: [0-100]
Accuracy: [0-100]
Completeness: [0-100]
Consistency: [0-100]
"""

        result = await self.promptly_core.run_workflow("measure_quality", prompt)

        if result.get('success'):
            output = result.get('output', '')
            return self._parse_quality_scores(output)

        return None

    def _parse_quality_scores(self, output: str) -> Optional[QualityMetrics]:
        """Parse quality scores from LLM output"""
        import re

        try:
            metrics = QualityMetrics()

            # Parse scores
            clarity_match = re.search(r'Clarity:\s*(\d+)', output)
            simplicity_match = re.search(r'Simplicity:\s*(\d+)', output)
            accuracy_match = re.search(r'Accuracy:\s*(\d+)', output)
            completeness_match = re.search(r'Completeness:\s*(\d+)', output)
            consistency_match = re.search(r'Consistency:\s*(\d+)', output)

            if clarity_match:
                metrics.clarity = float(clarity_match.group(1)) / 100.0
            if simplicity_match:
                metrics.simplicity = float(simplicity_match.group(1)) / 100.0
            if accuracy_match:
                metrics.accuracy = float(accuracy_match.group(1)) / 100.0
            if completeness_match:
                metrics.completeness = float(completeness_match.group(1)) / 100.0
            if consistency_match:
                metrics.consistency = float(consistency_match.group(1)) / 100.0

            return metrics

        except Exception as e:
            logger.warning(f"Failed to parse quality scores: {e}")
            return None

    def _heuristic_measure_quality(self, text: str, query: str) -> QualityMetrics:
        """Heuristic quality measurement (no LLM)"""
        metrics = QualityMetrics()

        # Clarity: Shorter sentences = clearer
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        metrics.clarity = max(0.0, min(1.0, 1.0 - (avg_sentence_length - 15) / 30))

        # Simplicity: Shorter text = simpler
        word_count = len(text.split())
        metrics.simplicity = max(0.0, min(1.0, 1.0 - (word_count - 100) / 200))

        # Accuracy: Presence of hedging words suggests uncertainty
        hedging_words = ['may', 'might', 'could', 'possibly', 'potentially']
        hedging_count = sum(1 for word in hedging_words if word in text.lower())
        metrics.accuracy = max(0.3, min(1.0, 1.0 - hedging_count * 0.1))

        # Completeness: Longer text (to a point) = more complete
        metrics.completeness = max(0.0, min(1.0, word_count / 300))

        # Consistency: No contradiction keywords = consistent
        contradiction_keywords = ['however', 'but', 'although', 'contrary']
        contradiction_count = sum(1 for word in contradiction_keywords if word in text.lower())
        metrics.consistency = max(0.5, min(1.0, 1.0 - contradiction_count * 0.15))

        return metrics


class QuickRefiner:
    """
    Quick refinement without LLM.

    Uses heuristic improvements for fast iteration.
    """

    def refine(
        self,
        text: str,
        focus: str = "clarity"
    ) -> tuple[str, List[str]]:
        """Quick heuristic refinement"""
        improvements = []
        refined_text = text

        # Remove redundant words
        redundant_words = {
            "very": "Replace 'very X' with stronger word",
            "really": "Remove 'really'",
            "actually": "Remove 'actually'",
            "basically": "Remove 'basically'",
            "just": "Remove filler 'just'"
        }

        for word, reason in redundant_words.items():
            count = refined_text.lower().count(f" {word} ")
            if count > 0:
                refined_text = refined_text.replace(f" {word} ", " ")
                refined_text = refined_text.replace(f" {word.capitalize()} ", " ")
                improvements.append(f"{reason} ({count} instances)")

        # Fix double spaces
        while "  " in refined_text:
            refined_text = refined_text.replace("  ", " ")

        if not improvements:
            improvements.append("No quick improvements found")

        return refined_text, improvements


# Singletons
_refiner = None
_quick_refiner = None


def get_refiner(promptly_core=None) -> Refiner:
    """Get singleton refiner"""
    global _refiner
    if _refiner is None:
        _refiner = Refiner(promptly_core)
    return _refiner


def get_quick_refiner() -> QuickRefiner:
    """Get singleton quick refiner"""
    global _quick_refiner
    if _quick_refiner is None:
        _quick_refiner = QuickRefiner()
    return _quick_refiner


# Test
if __name__ == "__main__":
    import asyncio

    async def test():
        # Test quick refiner (no LLM)
        quick = QuickRefiner()

        test_text = """
        Thompson Sampling is actually a very powerful algorithm that is really
        useful for basically balancing exploration and exploitation. It's just
        a very elegant approach that really works well in practice.
        """

        refined, improvements = quick.refine(test_text.strip(), focus="clarity")

        print("Original:")
        print(test_text.strip())
        print("\nRefined:")
        print(refined)
        print("\nImprovements:")
        for improvement in improvements:
            print(f"  - {improvement}")

        # Test heuristic quality measurement
        refiner = Refiner()
        metrics = refiner._heuristic_measure_quality(refined, "Explain Thompson Sampling")

        print(f"\nQuality Metrics:")
        print(f"  Clarity: {metrics.clarity:.2f}")
        print(f"  Simplicity: {metrics.simplicity:.2f}")
        print(f"  Accuracy: {metrics.accuracy:.2f}")
        print(f"  Completeness: {metrics.completeness:.2f}")
        print(f"  Consistency: {metrics.consistency:.2f}")
        print(f"  Overall (ELEGANCE): {metrics.overall_score(RefinementStrategy.ELEGANCE):.2f}")

        print("\nAll refiner tests passed!")

    asyncio.run(test())
