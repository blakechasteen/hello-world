"""Prompt Mutation Testing for robustness evaluation.

Implements systematic prompt mutations to test robustness and quality degradation
across different prompt variations (case changes, constraints, punctuation, etc).

Now with LLM-powered evaluation using Ollama (December 2025).

Status: ✅ Production Ready (November 2025)
Updated: December 2025 - LLMJudge integration
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Any
from HoloLoom.prompting.testing.protocol import PromptTestConfig, PromptTestResult

# LLM Judge integration (December 2025)
try:
    from HoloLoom.chaining.evaluation import (
        LLMJudge,
        JudgeCriteria,
        JudgeConfig,
        create_judge,
    )
    LLM_JUDGE_AVAILABLE = True
except ImportError:
    LLM_JUDGE_AVAILABLE = False
    LLMJudge = None
    JudgeCriteria = None
    JudgeConfig = None


class MutationType(Enum):
    """Types of mutations to apply to prompts."""

    CASE_LOWER = "case_lower"
    CASE_UPPER = "case_upper"
    ADD_CONSTRAINT = "add_constraint"
    ADD_CLARIFICATION = "add_clarification"
    PUNCTUATION_CHANGE = "punctuation_change"
    WORD_REORDER = "word_reorder"
    SYNONYM_REPLACE = "synonym_replace"
    TYPO_INJECT = "typo_inject"
    WHITESPACE_CHANGE = "whitespace_change"
    NEGATION_ADD = "negation_add"


@dataclass
class Mutation:
    """A single mutation applied to a prompt."""

    mutation_type: MutationType
    original: str
    mutated: str
    description: str


class PromptMutator:
    """Applies systematic mutations to prompts for robustness testing."""

    def __init__(self, enabled_mutations: Optional[List[MutationType]] = None):
        """Initialize mutator with optional mutation filter.

        Args:
            enabled_mutations: List of mutation types to enable (None = all)
        """
        self.enabled_mutations = (
            enabled_mutations if enabled_mutations else list(MutationType)
        )

    def mutate(self, prompt: str) -> List[Mutation]:
        """Apply all enabled mutations to a prompt.

        Args:
            prompt: Original prompt text

        Returns:
            List of Mutation objects with original and mutated versions
        """
        mutations = []
        for mutation_type in self.enabled_mutations:
            mutation = self._apply_mutation(prompt, mutation_type)
            if mutation:
                mutations.append(mutation)
        return mutations

    def _apply_mutation(
        self, prompt: str, mutation_type: MutationType
    ) -> Optional[Mutation]:
        """Apply a specific mutation to a prompt.

        Args:
            prompt: Original prompt text
            mutation_type: Type of mutation to apply

        Returns:
            Mutation object or None if mutation not applicable
        """
        if mutation_type == MutationType.CASE_LOWER:
            return self._lower(prompt)
        elif mutation_type == MutationType.CASE_UPPER:
            return self._upper(prompt)
        elif mutation_type == MutationType.ADD_CONSTRAINT:
            return self._add_constraint(prompt)
        elif mutation_type == MutationType.ADD_CLARIFICATION:
            return self._add_clarification(prompt)
        elif mutation_type == MutationType.PUNCTUATION_CHANGE:
            return self._punctuation_change(prompt)
        elif mutation_type == MutationType.WORD_REORDER:
            return self._word_reorder(prompt)
        elif mutation_type == MutationType.SYNONYM_REPLACE:
            return self._synonym_replace(prompt)
        elif mutation_type == MutationType.TYPO_INJECT:
            return self._typo_inject(prompt)
        elif mutation_type == MutationType.WHITESPACE_CHANGE:
            return self._whitespace_change(prompt)
        elif mutation_type == MutationType.NEGATION_ADD:
            return self._negation_add(prompt)
        return None

    def _lower(self, prompt: str) -> Mutation:
        """Convert prompt to lowercase."""
        mutated = prompt.lower()
        return Mutation(
            mutation_type=MutationType.CASE_LOWER,
            original=prompt,
            mutated=mutated,
            description="Convert to lowercase",
        )

    def _upper(self, prompt: str) -> Mutation:
        """Convert prompt to uppercase."""
        mutated = prompt.upper()
        return Mutation(
            mutation_type=MutationType.CASE_UPPER,
            original=prompt,
            mutated=mutated,
            description="Convert to uppercase",
        )

    def _add_constraint(self, prompt: str) -> Mutation:
        """Add brevity constraint to prompt."""
        constraint = " Please be brief."
        mutated = prompt.rstrip(".?!") + constraint
        return Mutation(
            mutation_type=MutationType.ADD_CONSTRAINT,
            original=prompt,
            mutated=mutated,
            description="Add brevity constraint",
        )

    def _add_clarification(self, prompt: str) -> Mutation:
        """Add clarification prefix to prompt."""
        clarification = "In simple terms, "
        mutated = clarification + prompt
        return Mutation(
            mutation_type=MutationType.ADD_CLARIFICATION,
            original=prompt,
            mutated=mutated,
            description="Add clarification prefix",
        )

    def _punctuation_change(self, prompt: str) -> Mutation:
        """Change ending punctuation."""
        mutated = prompt.rstrip(".?!")
        if mutated.endswith("?"):
            mutated = mutated[:-1] + "."
        else:
            mutated = mutated + "?"
        return Mutation(
            mutation_type=MutationType.PUNCTUATION_CHANGE,
            original=prompt,
            mutated=mutated,
            description="Change punctuation",
        )

    def _word_reorder(self, prompt: str) -> Mutation:
        """Reverse word order in prompt."""
        words = prompt.split()
        mutated = " ".join(reversed(words))
        return Mutation(
            mutation_type=MutationType.WORD_REORDER,
            original=prompt,
            mutated=mutated,
            description="Reverse word order",
        )

    def _synonym_replace(self, prompt: str) -> Mutation:
        """Replace common words with synonyms."""
        synonyms = {
            "explain": "describe",
            "what": "which",
            "help": "assist",
            "find": "locate",
            "show": "display",
        }
        mutated = prompt
        for word, synonym in synonyms.items():
            mutated = re.sub(rf"\b{word}\b", synonym, mutated, flags=re.IGNORECASE)
        return Mutation(
            mutation_type=MutationType.SYNONYM_REPLACE,
            original=prompt,
            mutated=mutated,
            description="Replace with synonyms",
        )

    def _typo_inject(self, prompt: str) -> Mutation:
        """Inject a typo into the prompt."""
        words = prompt.split()
        if words:
            target_word = words[0]
            if len(target_word) > 2:
                typo_word = target_word[:-1]  # Remove last character
                mutated = prompt.replace(target_word, typo_word, 1)
            else:
                mutated = prompt
        else:
            mutated = prompt
        return Mutation(
            mutation_type=MutationType.TYPO_INJECT,
            original=prompt,
            mutated=mutated,
            description="Inject typo",
        )

    def _whitespace_change(self, prompt: str) -> Mutation:
        """Add extra whitespace or change spacing."""
        mutated = re.sub(r"\s+", "  ", prompt)  # Double spaces
        return Mutation(
            mutation_type=MutationType.WHITESPACE_CHANGE,
            original=prompt,
            mutated=mutated,
            description="Add extra whitespace",
        )

    def _negation_add(self, prompt: str) -> Mutation:
        """Add negation to prompt."""
        negation = "Don't just "
        mutated = negation + prompt.lower()
        return Mutation(
            mutation_type=MutationType.NEGATION_ADD,
            original=prompt,
            mutated=mutated,
            description="Add negation",
        )


class MutationTester:
    """Tests prompt robustness to mutations.

    Now supports LLM-powered evaluation using Ollama (December 2025).
    Falls back to heuristic evaluation when LLM unavailable.
    """

    def __init__(
        self,
        mutator: Optional[PromptMutator] = None,
        quality_evaluator: Optional[Callable] = None,
        config: Optional[PromptTestConfig] = None,
        llm_judge: Optional[Any] = None,
    ):
        """Initialize mutation tester.

        Args:
            mutator: PromptMutator instance (created if not provided)
            quality_evaluator: Callable that evaluates prompt quality (optional if llm_judge provided)
            config: Optional test configuration
            llm_judge: Optional LLMJudge instance for LLM-based evaluation
        """
        self.mutator = mutator or PromptMutator()
        self.config = config or PromptTestConfig()
        self.baseline_quality = 0.0

        # LLM Judge integration (December 2025)
        self.llm_judge = llm_judge
        self._llm_judge_available = (
            LLM_JUDGE_AVAILABLE
            and llm_judge is not None
            and self.config.use_llm_judge
        )

        # Set quality evaluator - prefer LLM, fallback to provided or heuristic
        if quality_evaluator is not None:
            self.quality_evaluator = quality_evaluator
        elif self._llm_judge_available:
            self.quality_evaluator = self._evaluate_with_llm
        else:
            self.quality_evaluator = self._heuristic_evaluator

    async def _heuristic_evaluator(self, prompt: str) -> float:
        """Fallback heuristic quality evaluator.

        Uses simple metrics when LLM evaluation unavailable.
        """
        # Basic heuristics based on prompt characteristics
        score = 0.7  # Base score

        # Length check (reasonable length gets bonus)
        word_count = len(prompt.split())
        if 5 <= word_count <= 50:
            score += 0.1
        elif word_count < 3:
            score -= 0.2

        # Has question mark (indicates clear question)
        if "?" in prompt:
            score += 0.05

        # Has proper capitalization
        if prompt and prompt[0].isupper():
            score += 0.05

        # Not all caps (SHOUTING)
        if prompt.isupper() and len(prompt) > 10:
            score -= 0.1

        return min(1.0, max(0.0, score))

    async def _evaluate_with_llm(self, prompt: str) -> float:
        """Evaluate prompt quality using LLM Judge.

        Args:
            prompt: Prompt text to evaluate

        Returns:
            Quality score (0.0-1.0)
        """
        if not self._llm_judge_available:
            return await self._heuristic_evaluator(prompt)

        try:
            # Build criteria list from config
            criteria = []
            for criterion_name in self.config.llm_criteria:
                try:
                    criteria.append(JudgeCriteria[criterion_name.upper()])
                except KeyError:
                    continue

            # Default criteria if none valid
            if not criteria:
                criteria = [JudgeCriteria.QUALITY, JudgeCriteria.COHERENCE]

            # Create judge config
            judge_config = JudgeConfig(
                criteria=criteria,
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
            )

            # Evaluate the prompt (treating it as both task and output for mutation testing)
            result = await self.llm_judge.evaluate(
                task="Evaluate this prompt for quality and clarity",
                output=prompt,
                config=judge_config,
            )

            # Calculate average score across criteria
            if result.scores:
                total = sum(score.score for score in result.scores)
                return total / len(result.scores)
            return 0.5

        except Exception as e:
            # Fallback to heuristic on error
            if self.config.fallback_to_heuristic:
                return await self._heuristic_evaluator(prompt)
            raise

    def mutate(self, prompt: str) -> List[Mutation]:
        """Apply mutations to a prompt.

        Convenience method that delegates to internal mutator.

        Args:
            prompt: Original prompt text

        Returns:
            List of Mutation objects
        """
        return self.mutator.mutate(prompt)

    async def test_mutations(
        self, prompt: str, baseline_quality: float
    ) -> List[PromptTestResult]:
        """Test mutations against baseline quality.

        Args:
            prompt: Original prompt to mutate
            baseline_quality: Baseline quality score (0.0-1.0)

        Returns:
            List of test results for each mutation
        """
        self.baseline_quality = baseline_quality
        mutations = self.mutator.mutate(prompt)
        results = []

        for mutation in mutations:
            try:
                # Evaluate mutated prompt
                quality = await self.quality_evaluator(mutation.mutated)

                # Check for degradation
                degradation = baseline_quality - quality
                regressions = []

                if degradation > 0.2:  # >20% quality drop
                    regressions.append(
                        f"Quality degraded {degradation:.1%} on {mutation.mutation_type.value}"
                    )

                # Create result (basic version without full test case)
                result = PromptTestResult(
                    test_case=None,  # type: ignore
                    passed=degradation <= 0.2,
                    quality_scores={"quality": quality},
                    latency_ms=0.0,
                    token_count=0,
                    regressions_detected=regressions,
                )
                results.append(result)

            except Exception as e:
                results.append(
                    PromptTestResult(
                        test_case=None,  # type: ignore
                        passed=False,
                        quality_scores={},
                        latency_ms=0.0,
                        token_count=0,
                        error_message=str(e),
                    )
                )

        return results

    def calculate_robustness_score(
        self, results: List[PromptTestResult], baseline: float
    ) -> float:
        """Calculate overall robustness score from mutation test results.

        Args:
            results: List of mutation test results
            baseline: Baseline quality score

        Returns:
            Robustness score (0.0-1.0)
        """
        if not results:
            return 0.0

        passed = sum(1 for r in results if r.passed)
        robustness = passed / len(results)

        return robustness


def create_mutation_tester(
    config: Optional[PromptTestConfig] = None,
    llm_judge: Optional[Any] = None,
    enabled_mutations: Optional[List[MutationType]] = None,
) -> MutationTester:
    """Factory function to create a mutation tester.

    Creates a MutationTester with LLM-based evaluation when available,
    falling back to heuristics when LLM is unavailable.

    Args:
        config: Optional test configuration
        llm_judge: Optional pre-created LLMJudge instance
        enabled_mutations: Optional list of mutation types to enable

    Returns:
        MutationTester instance with appropriate evaluation backend
    """
    config = config or PromptTestConfig()
    mutator = PromptMutator(enabled_mutations=enabled_mutations)

    # Create LLM Judge if available and configured
    judge = llm_judge
    if judge is None and LLM_JUDGE_AVAILABLE and config.use_llm_judge:
        try:
            judge = create_judge(
                provider=config.llm_provider,
                model=config.llm_model,
            )
        except Exception:
            # Graceful degradation - will use heuristics
            judge = None

    return MutationTester(
        mutator=mutator,
        config=config,
        llm_judge=judge,
    )


async def create_mutation_tester_async(
    config: Optional[PromptTestConfig] = None,
    enabled_mutations: Optional[List[MutationType]] = None,
) -> MutationTester:
    """Async factory function to create a mutation tester.

    Useful when LLM Judge creation requires async initialization.

    Args:
        config: Optional test configuration
        enabled_mutations: Optional list of mutation types to enable

    Returns:
        MutationTester instance with appropriate evaluation backend
    """
    config = config or PromptTestConfig()
    mutator = PromptMutator(enabled_mutations=enabled_mutations)

    # Create LLM Judge if available and configured
    judge = None
    if LLM_JUDGE_AVAILABLE and config.use_llm_judge:
        try:
            judge = create_judge(
                provider=config.llm_provider,
                model=config.llm_model,
            )
        except Exception:
            # Graceful degradation - will use heuristics
            judge = None

    return MutationTester(
        mutator=mutator,
        config=config,
        llm_judge=judge,
    )
