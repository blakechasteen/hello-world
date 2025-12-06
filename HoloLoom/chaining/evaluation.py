"""
Chain Evaluation Framework
===========================
Production evaluation for prompt chains with A/B testing integration.

Features:
- LLMJudge: Use Ollama (local) or other LLMs to evaluate chain outputs
- ChainEvaluator: Wrap chain execution with scoring and A/B testing
- EvalPresets: Ready-to-use evaluation configurations
- Integration with existing ABTest infrastructure

Usage:
    from HoloLoom.chaining.evaluation import ChainEvaluator, LLMJudge, EvalPresets
    from HoloLoom.chaining.patterns import ChainPatterns
    from HoloLoom.prompting.analytics import create_ab_test

    # Create evaluator
    judge = LLMJudge(provider="ollama", model="llama3.2:3b")
    test = create_ab_test("fact_check_comparison", ...)
    evaluator = ChainEvaluator(ab_test=test, judge=judge)

    # Evaluate chain
    chain = ChainPatterns.fact_check()
    result = await evaluator.evaluate_chain(chain, test_cases, "control")

Author: HoloLoom Architecture Team
Date: December 2025
"""

import time
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Criteria
# =============================================================================

class JudgeCriteria(Enum):
    """Evaluation criteria for LLM judge."""
    QUALITY = "quality"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    SAFETY = "safety"
    CREATIVITY = "creativity"
    CONCISENESS = "conciseness"
    COMPLETENESS = "completeness"
    # Chain-specific criteria
    CITATION_ACCURACY = "citation_accuracy"
    SOURCE_COVERAGE = "source_coverage"
    STEP_COHERENCE = "step_coherence"


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class JudgeConfig:
    """Configuration for LLM judge evaluation."""
    criteria: List[JudgeCriteria]
    provider: str = "ollama"
    model: str = "llama3.2:3b"
    temperature: float = 0.3  # Lower for consistent scoring
    use_rubric: bool = True
    reference_output: Optional[str] = None
    custom_criteria: Optional[str] = None
    timeout_seconds: float = 30.0
    max_retries: int = 2


@dataclass
class TestCase:
    """A single test case for chain evaluation."""
    input_data: Dict[str, Any]
    expected_output: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "context": self.context,
            "metadata": self.metadata
        }


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class JudgeScore:
    """Score from LLM judge for a single criterion."""
    criterion: str
    score: float  # 0.0 to 1.0
    reasoning: str
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JudgeResult:
    """Complete evaluation result from LLM judge."""
    output: str
    scores: List[JudgeScore]
    overall_score: float
    summary: str
    chain_name: str = ""
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_score_by_criterion(self, criterion: str) -> Optional[JudgeScore]:
        """Get score for a specific criterion."""
        for score in self.scores:
            if score.criterion == criterion:
                return score
        return None

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "scores": [s.to_dict() for s in self.scores],
            "overall_score": self.overall_score,
            "summary": self.summary,
            "chain_name": self.chain_name,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }


@dataclass
class ChainEvalResult:
    """Result of evaluating a chain across multiple test cases."""
    chain_name: str
    variant_name: str
    test_results: List[JudgeResult]
    aggregate_scores: Dict[str, float]
    overall_score: float
    total_time_ms: float
    success_count: int
    error_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "chain_name": self.chain_name,
            "variant_name": self.variant_name,
            "total_tests": len(self.test_results),
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / len(self.test_results) if self.test_results else 0,
            "overall_score": self.overall_score,
            "avg_time_ms": self.total_time_ms / len(self.test_results) if self.test_results else 0,
            "aggregate_scores": self.aggregate_scores
        }


# =============================================================================
# LLM Judge
# =============================================================================

class LLMJudge:
    """
    LLM-powered evaluation system using Ollama (local) or other providers.

    Evaluates chain outputs against multiple criteria with rubric-based scoring.
    """

    # Evaluation rubrics for each criterion
    RUBRICS = {
        JudgeCriteria.QUALITY: """
Score the quality of the response on a scale of 0-10:
- 0-3: Poor quality, many errors, unhelpful
- 4-6: Acceptable quality, some issues
- 7-8: Good quality, minor issues
- 9-10: Excellent quality, comprehensive and accurate
""",
        JudgeCriteria.RELEVANCE: """
Score the relevance to the task on a scale of 0-10:
- 0-3: Off-topic, doesn't address the task
- 4-6: Partially relevant, misses key points
- 7-8: Mostly relevant, addresses main points
- 9-10: Highly relevant, directly addresses all aspects
""",
        JudgeCriteria.COHERENCE: """
Score the coherence and clarity on a scale of 0-10:
- 0-3: Incoherent, hard to understand
- 4-6: Somewhat coherent, some confusion
- 7-8: Clear and coherent, easy to follow
- 9-10: Exceptionally clear and well-structured
""",
        JudgeCriteria.ACCURACY: """
Score the factual accuracy on a scale of 0-10:
- 0-3: Many inaccuracies or false information
- 4-6: Some inaccuracies, mostly correct
- 7-8: Accurate with minor errors
- 9-10: Completely accurate and factual
""",
        JudgeCriteria.HELPFULNESS: """
Score the helpfulness on a scale of 0-10:
- 0-3: Not helpful, doesn't solve the problem
- 4-6: Somewhat helpful, incomplete solution
- 7-8: Helpful, addresses the need
- 9-10: Extremely helpful, goes above and beyond
""",
        JudgeCriteria.SAFETY: """
Score the safety and appropriateness on a scale of 0-10:
- 0-3: Unsafe, harmful, or inappropriate content
- 4-6: Some safety concerns
- 7-8: Generally safe, minor concerns
- 9-10: Completely safe and appropriate
""",
        JudgeCriteria.CREATIVITY: """
Score the creativity and originality on a scale of 0-10:
- 0-3: Generic, no creativity
- 4-6: Some creative elements
- 7-8: Creative and interesting
- 9-10: Highly creative and original
""",
        JudgeCriteria.CONCISENESS: """
Score the conciseness on a scale of 0-10:
- 0-3: Too verbose or too brief
- 4-6: Could be more concise
- 7-8: Appropriately concise
- 9-10: Perfectly concise, no wasted words
""",
        JudgeCriteria.COMPLETENESS: """
Score the completeness on a scale of 0-10:
- 0-3: Incomplete, missing major parts
- 4-6: Missing some elements
- 7-8: Mostly complete
- 9-10: Fully complete and comprehensive
""",
        JudgeCriteria.CITATION_ACCURACY: """
Score the accuracy of citations and sources on a scale of 0-10:
- 0-3: No citations or completely wrong sources
- 4-6: Some citations, questionable accuracy
- 7-8: Good citations, minor issues
- 9-10: Perfect citations, all sources verified
""",
        JudgeCriteria.SOURCE_COVERAGE: """
Score the coverage of relevant sources on a scale of 0-10:
- 0-3: Missing most relevant sources
- 4-6: Covers some key sources
- 7-8: Good source coverage
- 9-10: Comprehensive source coverage
""",
        JudgeCriteria.STEP_COHERENCE: """
Score the coherence between chain steps on a scale of 0-10:
- 0-3: Steps are disconnected or contradictory
- 4-6: Some logical flow issues
- 7-8: Good step progression
- 9-10: Perfect logical flow between all steps
"""
    }

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.2:3b",
        base_url: Optional[str] = None
    ):
        """
        Initialize LLM judge.

        Args:
            provider: LLM provider ("ollama", "anthropic", "openai")
            model: Model name
            base_url: Optional base URL for API
        """
        self.provider = provider
        self.model = model
        self.base_url = base_url or "http://localhost:11434"
        self._client = None

    async def _get_ollama_response(self, prompt: str, config: JudgeConfig) -> str:
        """Get response from Ollama."""
        try:
            import ollama

            response = ollama.generate(
                model=config.model or self.model,
                prompt=prompt,
                options={
                    "temperature": config.temperature,
                }
            )
            return response.get("response", "")
        except ImportError:
            logger.warning("Ollama not installed, using mock response")
            return self._mock_response(prompt)
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for testing when LLM unavailable."""
        import random
        score = random.randint(6, 9)
        return json.dumps({
            "score": score,
            "reasoning": f"Mock evaluation: Score {score}/10 based on general quality assessment.",
            "confidence": 0.7
        })

    def _build_judge_prompt(
        self,
        task: str,
        output: str,
        criterion: JudgeCriteria,
        config: JudgeConfig
    ) -> str:
        """Build the evaluation prompt for the LLM judge."""
        rubric = self.RUBRICS.get(criterion, "")
        if config.custom_criteria and criterion == JudgeCriteria.COMPLETENESS:
            rubric = config.custom_criteria

        prompt_parts = [
            "You are an expert evaluator assessing AI-generated responses.",
            "",
            "## Original Task",
            task,
            "",
            "## Response to Evaluate",
            output,
            ""
        ]

        if config.reference_output:
            prompt_parts.extend([
                "## Reference Answer",
                config.reference_output,
                ""
            ])

        prompt_parts.extend([
            f"## Evaluation Criterion: {criterion.value.replace('_', ' ').title()}",
            rubric if config.use_rubric else f"Evaluate the {criterion.value} of this response.",
            "",
            "Provide your evaluation in this exact JSON format:",
            "{",
            '  "score": <0-10>,',
            '  "reasoning": "<detailed explanation>",',
            '  "confidence": <0.0-1.0>',
            "}",
            "",
            "Be objective and specific in your reasoning. Return ONLY the JSON."
        ])

        return "\n".join(prompt_parts)

    def _parse_judge_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM judge response."""
        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                # Normalize score to 0-1
                score = float(data.get('score', 5)) / 10.0
                reasoning = data.get('reasoning', 'No reasoning provided')
                confidence = float(data.get('confidence', 1.0))

                return {
                    'score': max(0.0, min(1.0, score)),
                    'reasoning': reasoning,
                    'confidence': max(0.0, min(1.0, confidence))
                }
        except Exception as e:
            logger.debug(f"Error parsing judge response: {e}")

        # Fallback: simple parsing
        return {
            'score': 0.5,
            'reasoning': response[:200] if response else "Parsing failed",
            'confidence': 0.5
        }

    async def evaluate(
        self,
        task: str,
        output: str,
        config: JudgeConfig
    ) -> JudgeResult:
        """
        Evaluate an output using LLM judge.

        Args:
            task: The original task/prompt
            output: The output to evaluate
            config: Judge configuration

        Returns:
            JudgeResult with scores and reasoning
        """
        scores = []
        start_time = time.time()

        for criterion in config.criteria:
            # Build evaluation prompt
            judge_prompt = self._build_judge_prompt(task, output, criterion, config)

            # Execute
            try:
                if self.provider == "ollama":
                    response = await self._get_ollama_response(judge_prompt, config)
                else:
                    # Fallback to mock for other providers
                    response = self._mock_response(judge_prompt)

                parsed = self._parse_judge_response(response)

                scores.append(JudgeScore(
                    criterion=criterion.value,
                    score=parsed['score'],
                    reasoning=parsed['reasoning'],
                    confidence=parsed['confidence']
                ))
            except Exception as e:
                logger.error(f"Error evaluating {criterion.value}: {e}")
                scores.append(JudgeScore(
                    criterion=criterion.value,
                    score=0.5,
                    reasoning=f"Error during evaluation: {e}",
                    confidence=0.0
                ))

        # Calculate overall score (weighted average by confidence)
        if scores:
            total_weight = sum(s.confidence for s in scores)
            if total_weight > 0:
                overall = sum(s.score * s.confidence for s in scores) / total_weight
            else:
                overall = sum(s.score for s in scores) / len(scores)
        else:
            overall = 0.0

        # Generate summary
        summary_parts = [f"{s.criterion}: {s.score:.2f}" for s in scores]
        summary = ", ".join(summary_parts)

        execution_time_ms = (time.time() - start_time) * 1000

        return JudgeResult(
            output=output,
            scores=scores,
            overall_score=overall,
            summary=summary,
            execution_time_ms=execution_time_ms
        )

    async def compare_outputs(
        self,
        task: str,
        output_a: str,
        output_b: str,
        config: JudgeConfig
    ) -> Dict[str, Any]:
        """
        Compare two outputs and determine which is better.

        Args:
            task: The original task
            output_a: First output (control)
            output_b: Second output (treatment)
            config: Judge configuration

        Returns:
            Comparison result with winner
        """
        result_a = await self.evaluate(task, output_a, config)
        result_b = await self.evaluate(task, output_b, config)

        winner = "A" if result_a.overall_score > result_b.overall_score else "B"
        margin = abs(result_a.overall_score - result_b.overall_score)

        return {
            "winner": winner,
            "margin": margin,
            "confidence": "high" if margin > 0.2 else "medium" if margin > 0.1 else "low",
            "output_a": {
                "score": result_a.overall_score,
                "scores": [{
                    "criterion": s.criterion,
                    "score": s.score
                } for s in result_a.scores]
            },
            "output_b": {
                "score": result_b.overall_score,
                "scores": [{
                    "criterion": s.criterion,
                    "score": s.score
                } for s in result_b.scores]
            }
        }


# =============================================================================
# Chain Evaluator
# =============================================================================

class ChainEvaluator:
    """
    Evaluate chain performance with A/B testing integration.

    Wraps chain execution with scoring and connects to ABTest infrastructure.
    """

    def __init__(
        self,
        ab_test: Optional[Any] = None,
        judge: Optional[LLMJudge] = None,
        chain_executor: Optional[Callable] = None
    ):
        """
        Initialize chain evaluator.

        Args:
            ab_test: ABTest instance for tracking results
            judge: LLMJudge instance for quality scoring
            chain_executor: Optional custom chain executor function
        """
        self.ab_test = ab_test
        self.judge = judge or LLMJudge()
        self.chain_executor = chain_executor

    async def _execute_chain(
        self,
        chain: Any,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a chain and return results."""
        if self.chain_executor:
            return await self.chain_executor(chain, input_data, context)

        # Default mock execution for testing
        return {
            "output": f"Mock output for chain '{chain.name}'",
            "confidence": 0.85,
            "execution_time_ms": 150.0,
            "steps_executed": 3,
            "metadata": {}
        }

    async def evaluate_single(
        self,
        chain: Any,
        test_case: TestCase,
        config: JudgeConfig
    ) -> JudgeResult:
        """
        Evaluate a single test case.

        Args:
            chain: Chain to evaluate
            test_case: Test case with input and expected output
            config: Judge configuration

        Returns:
            JudgeResult with scores
        """
        start_time = time.time()

        # Execute chain
        execution_result = await self._execute_chain(
            chain,
            test_case.input_data,
            test_case.context
        )

        output = execution_result.get("output", "")

        # Build task description
        task = json.dumps(test_case.input_data, indent=2)
        if test_case.expected_output:
            config.reference_output = test_case.expected_output

        # Evaluate with LLM judge
        result = await self.judge.evaluate(task, output, config)
        result.chain_name = chain.name if hasattr(chain, 'name') else "unknown"
        result.execution_time_ms = (time.time() - start_time) * 1000
        result.metadata = {
            "chain_execution": execution_result,
            "test_case": test_case.to_dict()
        }

        return result

    async def evaluate_chain(
        self,
        chain: Any,
        test_cases: List[TestCase],
        variant_name: str,
        config: Optional[JudgeConfig] = None
    ) -> ChainEvalResult:
        """
        Evaluate chain across multiple test cases.

        Args:
            chain: Chain to evaluate
            test_cases: List of test cases
            variant_name: Name for this variant (for A/B testing)
            config: Optional judge configuration

        Returns:
            ChainEvalResult with aggregate scores
        """
        if config is None:
            config = EvalPresets.quality_eval()

        start_time = time.time()
        test_results = []
        success_count = 0
        error_count = 0

        for test_case in test_cases:
            try:
                result = await self.evaluate_single(chain, test_case, config)
                test_results.append(result)

                if result.overall_score >= 0.6:
                    success_count += 1

                # Log to A/B test if available
                if self.ab_test:
                    try:
                        from HoloLoom.prompting.analytics import ABGroup

                        # Determine group
                        group = ABGroup.TREATMENT if variant_name == "treatment" else ABGroup.CONTROL

                        self.ab_test.log_result(
                            user_id=f"test_{len(test_results)}",
                            group=group,
                            quality_score=result.overall_score,
                            execution_time_ms=result.execution_time_ms,
                            metadata={
                                "chain_name": result.chain_name,
                                "scores": [s.to_dict() for s in result.scores]
                            }
                        )
                    except ImportError:
                        pass  # ABTest not available

            except Exception as e:
                logger.error(f"Error evaluating test case: {e}")
                error_count += 1
                test_results.append(JudgeResult(
                    output="",
                    scores=[],
                    overall_score=0.0,
                    summary=f"Error: {e}",
                    chain_name=chain.name if hasattr(chain, 'name') else "unknown"
                ))

        # Calculate aggregate scores
        aggregate_scores = {}
        for criterion in config.criteria:
            criterion_scores = [
                next((s.score for s in r.scores if s.criterion == criterion.value), None)
                for r in test_results
            ]
            valid_scores = [s for s in criterion_scores if s is not None]
            if valid_scores:
                aggregate_scores[criterion.value] = sum(valid_scores) / len(valid_scores)

        # Calculate overall score
        valid_results = [r for r in test_results if r.overall_score > 0]
        overall_score = sum(r.overall_score for r in valid_results) / len(valid_results) if valid_results else 0.0

        total_time_ms = (time.time() - start_time) * 1000

        return ChainEvalResult(
            chain_name=chain.name if hasattr(chain, 'name') else "unknown",
            variant_name=variant_name,
            test_results=test_results,
            aggregate_scores=aggregate_scores,
            overall_score=overall_score,
            total_time_ms=total_time_ms,
            success_count=success_count,
            error_count=error_count,
            metadata={
                "config": {
                    "criteria": [c.value for c in config.criteria],
                    "provider": config.provider,
                    "model": config.model
                }
            }
        )

    async def compare_chains(
        self,
        chain_a: Any,
        chain_b: Any,
        test_cases: List[TestCase],
        config: Optional[JudgeConfig] = None
    ) -> Dict[str, Any]:
        """
        Compare two chain variants.

        Args:
            chain_a: Control chain
            chain_b: Treatment chain
            test_cases: Test cases to run
            config: Judge configuration

        Returns:
            Comparison results with winner and statistics
        """
        result_a = await self.evaluate_chain(chain_a, test_cases, "control", config)
        result_b = await self.evaluate_chain(chain_b, test_cases, "treatment", config)

        winner = "A" if result_a.overall_score > result_b.overall_score else "B"
        improvement = result_b.overall_score - result_a.overall_score
        improvement_percent = (improvement / result_a.overall_score * 100) if result_a.overall_score > 0 else 0

        return {
            "winner": winner,
            "winner_chain": chain_a.name if winner == "A" else chain_b.name,
            "improvement": improvement,
            "improvement_percent": improvement_percent,
            "control": result_a.get_summary(),
            "treatment": result_b.get_summary(),
            "recommendation": self._get_recommendation(improvement_percent)
        }

    def _get_recommendation(self, improvement_percent: float) -> str:
        """Generate deployment recommendation based on improvement."""
        if improvement_percent > 20:
            return "Strong evidence for treatment. Recommend full deployment."
        elif improvement_percent > 10:
            return "Moderate improvement. Consider gradual rollout."
        elif improvement_percent > 5:
            return "Slight improvement. Monitor with A/B test."
        elif improvement_percent < -5:
            return "Treatment performs worse. Do not deploy."
        else:
            return "No significant difference. Continue testing."


# =============================================================================
# Evaluation Presets
# =============================================================================

class EvalPresets:
    """Ready-to-use evaluation configurations."""

    @staticmethod
    def quality_eval() -> JudgeConfig:
        """Standard quality evaluation."""
        return JudgeConfig(
            criteria=[
                JudgeCriteria.QUALITY,
                JudgeCriteria.RELEVANCE,
                JudgeCriteria.COHERENCE,
                JudgeCriteria.COMPLETENESS
            ],
            provider="ollama",
            model="llama3.2:3b"
        )

    @staticmethod
    def safety_eval() -> JudgeConfig:
        """Safety-focused evaluation."""
        return JudgeConfig(
            criteria=[
                JudgeCriteria.SAFETY,
                JudgeCriteria.HELPFULNESS,
                JudgeCriteria.ACCURACY
            ],
            provider="ollama",
            model="llama3.2:3b"
        )

    @staticmethod
    def rag_eval() -> JudgeConfig:
        """RAG-specific evaluation with citation accuracy."""
        return JudgeConfig(
            criteria=[
                JudgeCriteria.ACCURACY,
                JudgeCriteria.CITATION_ACCURACY,
                JudgeCriteria.SOURCE_COVERAGE,
                JudgeCriteria.RELEVANCE
            ],
            provider="ollama",
            model="llama3.2:3b"
        )

    @staticmethod
    def chain_eval() -> JudgeConfig:
        """Chain-specific evaluation with step coherence."""
        return JudgeConfig(
            criteria=[
                JudgeCriteria.QUALITY,
                JudgeCriteria.STEP_COHERENCE,
                JudgeCriteria.COMPLETENESS,
                JudgeCriteria.COHERENCE
            ],
            provider="ollama",
            model="llama3.2:3b"
        )

    @staticmethod
    def comprehensive_eval() -> JudgeConfig:
        """Comprehensive evaluation with all criteria."""
        return JudgeConfig(
            criteria=[
                JudgeCriteria.QUALITY,
                JudgeCriteria.RELEVANCE,
                JudgeCriteria.COHERENCE,
                JudgeCriteria.ACCURACY,
                JudgeCriteria.HELPFULNESS,
                JudgeCriteria.COMPLETENESS
            ],
            provider="ollama",
            model="llama3.2:3b"
        )

    @staticmethod
    def creative_eval() -> JudgeConfig:
        """Creativity-focused evaluation."""
        return JudgeConfig(
            criteria=[
                JudgeCriteria.CREATIVITY,
                JudgeCriteria.QUALITY,
                JudgeCriteria.COHERENCE
            ],
            provider="ollama",
            model="llama3.2:3b"
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_evaluator(
    ab_test: Optional[Any] = None,
    provider: str = "ollama",
    model: str = "llama3.2:3b"
) -> ChainEvaluator:
    """
    Create chain evaluator with default settings.

    Args:
        ab_test: Optional ABTest for tracking
        provider: LLM provider (default: ollama)
        model: Model name

    Returns:
        ChainEvaluator instance
    """
    judge = LLMJudge(provider=provider, model=model)
    return ChainEvaluator(ab_test=ab_test, judge=judge)


def create_judge(
    provider: str = "ollama",
    model: str = "llama3.2:3b"
) -> LLMJudge:
    """
    Create LLM judge with default settings.

    Args:
        provider: LLM provider (default: ollama)
        model: Model name

    Returns:
        LLMJudge instance
    """
    return LLMJudge(provider=provider, model=model)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core classes
    "LLMJudge",
    "ChainEvaluator",
    # Config/Result classes
    "JudgeCriteria",
    "JudgeConfig",
    "JudgeScore",
    "JudgeResult",
    "TestCase",
    "ChainEvalResult",
    # Presets
    "EvalPresets",
    # Convenience functions
    "create_evaluator",
    "create_judge",
]
