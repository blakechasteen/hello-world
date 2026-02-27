"""
Model Extension SDK Benchmark Utilities

Compare memory-augmented LLMs against vanilla LLMs to quantify:
- Answer quality improvements
- Knowledge retention over sessions
- Token efficiency (context reuse)
- Latency overhead
- Hallucination reduction

Usage:
    from HoloLoom.model_extension import MemoryAugmentedLLM
    from HoloLoom.model_extension.benchmark import (
        BenchmarkSuite,
        QualityBenchmark,
        EfficiencyBenchmark,
        run_benchmark_suite,
    )

    # Run full benchmark suite
    async with MemoryAugmentedLLM(provider="anthropic") as llm:
        results = await run_benchmark_suite(llm, dataset="factual_qa")
        print(results.summary())

    # Or run individual benchmarks
    quality = QualityBenchmark()
    results = await quality.run(llm, questions=my_questions)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Callable, Tuple
import asyncio
import time
import statistics
import json


class BenchmarkType(Enum):
    """Types of benchmarks available."""
    QUALITY = "quality"              # Answer quality comparison
    EFFICIENCY = "efficiency"        # Token usage and latency
    RETENTION = "retention"          # Knowledge retention over time
    HALLUCINATION = "hallucination"  # Hallucination detection
    DOMAIN = "domain"                # Domain-specific performance


@dataclass
class BenchmarkQuestion:
    """A single question for benchmarking."""
    question: str
    expected_answer: Optional[str] = None
    domain: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    requires_memory: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark question."""
    question: str
    augmented_answer: str
    vanilla_answer: Optional[str] = None
    augmented_confidence: float = 0.0
    vanilla_confidence: float = 0.0
    augmented_latency_ms: float = 0.0
    vanilla_latency_ms: float = 0.0
    augmented_sources: int = 0
    quality_score: float = 0.0  # 0.0-1.0, how good was the answer
    improvement_score: float = 0.0  # How much better than vanilla
    used_memory: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    benchmark_type: BenchmarkType
    total_questions: int
    avg_quality_score: float
    avg_improvement: float
    avg_augmented_latency_ms: float
    avg_vanilla_latency_ms: float
    latency_overhead_percent: float
    memory_utilization_percent: float
    avg_sources_per_query: float
    avg_confidence_augmented: float
    avg_confidence_vanilla: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "benchmark_type": self.benchmark_type.value,
            "total_questions": self.total_questions,
            "quality": {
                "avg_score": self.avg_quality_score,
                "avg_improvement": self.avg_improvement,
            },
            "performance": {
                "augmented_latency_ms": self.avg_augmented_latency_ms,
                "vanilla_latency_ms": self.avg_vanilla_latency_ms,
                "overhead_percent": self.latency_overhead_percent,
            },
            "memory": {
                "utilization_percent": self.memory_utilization_percent,
                "avg_sources_per_query": self.avg_sources_per_query,
            },
            "confidence": {
                "augmented": self.avg_confidence_augmented,
                "vanilla": self.avg_confidence_vanilla,
            },
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"{'='*60}",
            f"BENCHMARK SUMMARY: {self.benchmark_type.value.upper()}",
            f"{'='*60}",
            f"",
            f"Questions Tested: {self.total_questions}",
            f"",
            f"QUALITY METRICS:",
            f"  Average Quality Score:  {self.avg_quality_score:.2f} / 1.0",
            f"  Improvement vs Vanilla: {self.avg_improvement:+.1%}",
            f"",
            f"PERFORMANCE METRICS:",
            f"  Augmented Latency:      {self.avg_augmented_latency_ms:.1f}ms",
            f"  Vanilla Latency:        {self.avg_vanilla_latency_ms:.1f}ms",
            f"  Overhead:               {self.latency_overhead_percent:+.1f}%",
            f"",
            f"MEMORY UTILIZATION:",
            f"  Queries Using Memory:   {self.memory_utilization_percent:.1f}%",
            f"  Avg Sources per Query:  {self.avg_sources_per_query:.1f}",
            f"",
            f"CONFIDENCE:",
            f"  Augmented Confidence:   {self.avg_confidence_augmented:.2f}",
            f"  Vanilla Confidence:     {self.avg_confidence_vanilla:.2f}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


class BaseBenchmark:
    """Base class for all benchmarks."""

    benchmark_type: BenchmarkType = BenchmarkType.QUALITY

    def __init__(
        self,
        quality_scorer: Optional[Callable[[str, str, str], float]] = None,
    ):
        """
        Initialize benchmark.

        Args:
            quality_scorer: Optional function(question, answer, expected) -> 0.0-1.0
        """
        self.quality_scorer = quality_scorer or self._default_quality_scorer

    def _default_quality_scorer(
        self,
        question: str,
        answer: str,
        expected: Optional[str],
    ) -> float:
        """
        Default quality scoring based on heuristics.

        A proper implementation would use an LLM-as-judge or embedding similarity.
        """
        if not answer:
            return 0.0

        score = 0.5  # Base score for providing an answer

        # Length heuristic (not too short, not too long)
        words = len(answer.split())
        if 10 <= words <= 200:
            score += 0.1
        elif words < 5:
            score -= 0.2

        # Check for uncertainty markers (lower score if very uncertain)
        uncertainty_markers = [
            "i don't know",
            "i'm not sure",
            "cannot determine",
            "unclear",
        ]
        if any(marker in answer.lower() for marker in uncertainty_markers):
            score -= 0.1

        # If we have expected answer, check for keyword overlap
        if expected:
            expected_words = set(expected.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(expected_words & answer_words) / max(len(expected_words), 1)
            score += overlap * 0.3

        return max(0.0, min(1.0, score))

    async def run(
        self,
        llm,
        questions: List[BenchmarkQuestion],
        compare_vanilla: bool = True,
    ) -> Tuple[List[BenchmarkResult], BenchmarkSummary]:
        """
        Run benchmark on list of questions.

        Args:
            llm: MemoryAugmentedLLM instance
            questions: Questions to benchmark
            compare_vanilla: Whether to also test without memory

        Returns:
            Tuple of (individual results, summary)
        """
        results = []

        for q in questions:
            result = await self._benchmark_single(llm, q, compare_vanilla)
            results.append(result)

        summary = self._compute_summary(results)
        return results, summary

    async def _benchmark_single(
        self,
        llm,
        question: BenchmarkQuestion,
        compare_vanilla: bool,
    ) -> BenchmarkResult:
        """Benchmark a single question."""
        # Augmented query
        start = time.perf_counter()
        try:
            response = await llm.query(question.question)
            augmented_answer = response.answer
            augmented_confidence = response.confidence
            augmented_sources = len(response.sources)
        except Exception as e:
            augmented_answer = f"Error: {e}"
            augmented_confidence = 0.0
            augmented_sources = 0
        augmented_latency = (time.perf_counter() - start) * 1000

        # Vanilla query (bypass memory)
        vanilla_answer = None
        vanilla_confidence = 0.0
        vanilla_latency = 0.0

        if compare_vanilla:
            start = time.perf_counter()
            try:
                # Query with empty context to simulate vanilla
                vanilla_response = await llm.query(
                    question.question,
                    context_override="",  # No memory context
                )
                vanilla_answer = vanilla_response.answer
                vanilla_confidence = vanilla_response.confidence
            except Exception as e:
                vanilla_answer = f"Error: {e}"
                vanilla_confidence = 0.0
            vanilla_latency = (time.perf_counter() - start) * 1000

        # Score quality
        quality_score = self.quality_scorer(
            question.question,
            augmented_answer,
            question.expected_answer,
        )

        # Compute improvement
        if vanilla_answer:
            vanilla_quality = self.quality_scorer(
                question.question,
                vanilla_answer,
                question.expected_answer,
            )
            improvement = quality_score - vanilla_quality
        else:
            improvement = 0.0

        return BenchmarkResult(
            question=question.question,
            augmented_answer=augmented_answer,
            vanilla_answer=vanilla_answer,
            augmented_confidence=augmented_confidence,
            vanilla_confidence=vanilla_confidence,
            augmented_latency_ms=augmented_latency,
            vanilla_latency_ms=vanilla_latency,
            augmented_sources=augmented_sources,
            quality_score=quality_score,
            improvement_score=improvement,
            used_memory=augmented_sources > 0,
            metadata=question.metadata,
        )

    def _compute_summary(self, results: List[BenchmarkResult]) -> BenchmarkSummary:
        """Compute summary statistics from results."""
        if not results:
            return BenchmarkSummary(
                benchmark_type=self.benchmark_type,
                total_questions=0,
                avg_quality_score=0.0,
                avg_improvement=0.0,
                avg_augmented_latency_ms=0.0,
                avg_vanilla_latency_ms=0.0,
                latency_overhead_percent=0.0,
                memory_utilization_percent=0.0,
                avg_sources_per_query=0.0,
                avg_confidence_augmented=0.0,
                avg_confidence_vanilla=0.0,
            )

        # Compute averages
        avg_quality = statistics.mean(r.quality_score for r in results)
        avg_improvement = statistics.mean(r.improvement_score for r in results)
        avg_aug_latency = statistics.mean(r.augmented_latency_ms for r in results)
        avg_van_latency = statistics.mean(r.vanilla_latency_ms for r in results) if results[0].vanilla_answer else 0.0

        # Latency overhead
        if avg_van_latency > 0:
            overhead = ((avg_aug_latency - avg_van_latency) / avg_van_latency) * 100
        else:
            overhead = 0.0

        # Memory utilization
        memory_used = sum(1 for r in results if r.used_memory)
        memory_util = (memory_used / len(results)) * 100

        # Average sources
        avg_sources = statistics.mean(r.augmented_sources for r in results)

        # Confidence
        avg_conf_aug = statistics.mean(r.augmented_confidence for r in results)
        avg_conf_van = statistics.mean(r.vanilla_confidence for r in results) if results[0].vanilla_answer else 0.0

        return BenchmarkSummary(
            benchmark_type=self.benchmark_type,
            total_questions=len(results),
            avg_quality_score=avg_quality,
            avg_improvement=avg_improvement,
            avg_augmented_latency_ms=avg_aug_latency,
            avg_vanilla_latency_ms=avg_van_latency,
            latency_overhead_percent=overhead,
            memory_utilization_percent=memory_util,
            avg_sources_per_query=avg_sources,
            avg_confidence_augmented=avg_conf_aug,
            avg_confidence_vanilla=avg_conf_van,
        )


class QualityBenchmark(BaseBenchmark):
    """
    Benchmark focusing on answer quality.

    Tests whether memory augmentation improves answer accuracy,
    completeness, and relevance.
    """

    benchmark_type = BenchmarkType.QUALITY


class EfficiencyBenchmark(BaseBenchmark):
    """
    Benchmark focusing on efficiency metrics.

    Tests token usage, latency, and resource utilization.
    """

    benchmark_type = BenchmarkType.EFFICIENCY

    async def run_with_token_tracking(
        self,
        llm,
        questions: List[BenchmarkQuestion],
    ) -> Tuple[List[BenchmarkResult], BenchmarkSummary, Dict[str, int]]:
        """
        Run benchmark with detailed token tracking.

        Returns additional token usage statistics.
        """
        results, summary = await self.run(llm, questions)

        # Token tracking (would integrate with LLM provider's usage tracking)
        token_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "avg_context_tokens": 0,
            "memory_context_tokens": 0,
        }

        # TODO: Integrate with LLM provider's token counting
        # This would require changes to the wrapper to track token usage

        return results, summary, token_stats


class RetentionBenchmark(BaseBenchmark):
    """
    Benchmark focusing on knowledge retention.

    Tests whether the system remembers information across queries.
    """

    benchmark_type = BenchmarkType.RETENTION

    async def run_retention_test(
        self,
        llm,
        teach_facts: List[Tuple[str, str]],  # (fact, verification_question)
        delay_seconds: float = 0.0,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Test retention of taught facts.

        Args:
            llm: MemoryAugmentedLLM instance
            teach_facts: List of (fact_to_teach, question_to_verify)
            delay_seconds: Delay between teaching and testing

        Returns:
            Tuple of (retention_rate, detailed_results)
        """
        # Phase 1: Teach facts
        for fact, _ in teach_facts:
            await llm.learn(fact)

        # Optional delay
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        # Phase 2: Test retention
        detailed_results = []
        retained = 0

        for fact, verification_question in teach_facts:
            response = await llm.query(verification_question)

            # Check if the fact is reflected in the answer
            # (Simple heuristic - check for key terms)
            fact_words = set(fact.lower().split())
            answer_words = set(response.answer.lower().split())
            overlap = len(fact_words & answer_words) / len(fact_words)

            is_retained = overlap > 0.3 or bool(response.sources)  # Has sources from memory

            detailed_results.append({
                "fact": fact,
                "question": verification_question,
                "answer": response.answer,
                "retained": is_retained,
                "overlap_score": overlap,
                "sources_found": len(response.sources),
                "confidence": response.confidence,
            })

            if is_retained:
                retained += 1

        retention_rate = retained / len(teach_facts) if teach_facts else 0.0
        return retention_rate, detailed_results


class HallucinationBenchmark(BaseBenchmark):
    """
    Benchmark focusing on hallucination detection.

    Tests whether memory augmentation reduces hallucinations.
    """

    benchmark_type = BenchmarkType.HALLUCINATION

    async def run_hallucination_test(
        self,
        llm,
        questions: List[BenchmarkQuestion],
        hallucination_detector: Optional[Callable[[str, str], float]] = None,
    ) -> Tuple[float, float, List[Dict[str, Any]]]:
        """
        Test hallucination rates.

        Args:
            llm: MemoryAugmentedLLM instance
            questions: Questions to test
            hallucination_detector: Optional function(question, answer) -> hallucination_score

        Returns:
            Tuple of (augmented_rate, vanilla_rate, detailed_results)
        """
        detector = hallucination_detector or self._default_hallucination_detector
        detailed_results = []
        aug_hallucinations = 0
        van_hallucinations = 0

        for q in questions:
            # Augmented response
            aug_response = await llm.query(q.question)
            aug_score = detector(q.question, aug_response.answer)

            # Vanilla response (no memory)
            van_response = await llm.query(q.question, context_override="")
            van_score = detector(q.question, van_response.answer)

            # Track results
            if aug_score > 0.5:
                aug_hallucinations += 1
            if van_score > 0.5:
                van_hallucinations += 1

            detailed_results.append({
                "question": q.question,
                "augmented_answer": aug_response.answer,
                "vanilla_answer": van_response.answer,
                "augmented_hallucination_score": aug_score,
                "vanilla_hallucination_score": van_score,
                "augmented_confidence": aug_response.confidence,
                "augmented_sources": len(aug_response.sources),
            })

        n = len(questions) if questions else 1
        aug_rate = aug_hallucinations / n
        van_rate = van_hallucinations / n

        return aug_rate, van_rate, detailed_results

    def _default_hallucination_detector(
        self,
        question: str,
        answer: str,
    ) -> float:
        """
        Default hallucination detection based on heuristics.

        Returns score 0.0 (no hallucination) to 1.0 (definite hallucination).
        A proper implementation would use an LLM-as-judge or fact-checking system.
        """
        score = 0.0

        # Check for overconfident false claims
        overconfident_markers = [
            "definitely",
            "certainly",
            "without a doubt",
            "100%",
            "always",
            "never",
        ]

        for marker in overconfident_markers:
            if marker in answer.lower():
                score += 0.1

        # Check for specific numeric claims (often hallucinated)
        import re
        specific_numbers = re.findall(r'\b\d{4,}\b', answer)  # 4+ digit numbers
        score += len(specific_numbers) * 0.1

        # Check for proper nouns that might be invented
        # (Simplified heuristic - count capitalized words)
        words = answer.split()
        proper_nouns = sum(1 for w in words if w and w[0].isupper() and len(w) > 1)
        if proper_nouns > len(words) * 0.3:  # Too many proper nouns
            score += 0.2

        return min(1.0, score)


class DomainBenchmark(BaseBenchmark):
    """
    Benchmark focusing on domain-specific performance.

    Tests performance on specialized domain knowledge.
    """

    benchmark_type = BenchmarkType.DOMAIN

    def __init__(
        self,
        domain: str,
        domain_knowledge: List[str],
        **kwargs,
    ):
        """
        Initialize domain benchmark.

        Args:
            domain: Domain name (e.g., "medical", "legal", "technical")
            domain_knowledge: Domain knowledge to ingest
        """
        super().__init__(**kwargs)
        self.domain = domain
        self.domain_knowledge = domain_knowledge

    async def run_domain_test(
        self,
        llm,
        questions: List[BenchmarkQuestion],
    ) -> Tuple[float, float, BenchmarkSummary]:
        """
        Run domain-specific benchmark.

        First ingests domain knowledge, then tests on domain questions.

        Returns:
            Tuple of (before_ingestion_score, after_ingestion_score, summary)
        """
        # Test before ingestion
        before_results = []
        for q in questions:
            response = await llm.query(q.question)
            score = self.quality_scorer(q.question, response.answer, q.expected_answer)
            before_results.append(score)

        before_avg = statistics.mean(before_results) if before_results else 0.0

        # Ingest domain knowledge
        for knowledge in self.domain_knowledge:
            await llm.learn(knowledge, metadata={"domain": self.domain})

        # Test after ingestion
        results, summary = await self.run(llm, questions, compare_vanilla=False)
        after_avg = summary.avg_quality_score

        return before_avg, after_avg, summary


# =============================================================================
# Standard Benchmark Datasets
# =============================================================================

def get_factual_qa_dataset() -> List[BenchmarkQuestion]:
    """Get standard factual Q&A benchmark questions."""
    return [
        BenchmarkQuestion(
            question="What is machine learning?",
            expected_answer="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            domain="technology",
            difficulty="easy",
        ),
        BenchmarkQuestion(
            question="Explain the concept of recursion in programming.",
            expected_answer="Recursion is when a function calls itself to solve smaller instances of a problem.",
            domain="programming",
            difficulty="medium",
        ),
        BenchmarkQuestion(
            question="What are the main differences between SQL and NoSQL databases?",
            expected_answer="SQL databases are relational with structured schemas, while NoSQL are non-relational with flexible schemas.",
            domain="databases",
            difficulty="medium",
        ),
        BenchmarkQuestion(
            question="How does a neural network learn?",
            expected_answer="Neural networks learn through backpropagation, adjusting weights based on error gradients.",
            domain="machine_learning",
            difficulty="hard",
        ),
        BenchmarkQuestion(
            question="What is the purpose of version control?",
            expected_answer="Version control tracks changes to code, enables collaboration, and allows reverting to previous states.",
            domain="software_engineering",
            difficulty="easy",
        ),
    ]


def get_retention_dataset() -> List[Tuple[str, str]]:
    """Get standard retention test dataset."""
    return [
        (
            "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation.",
            "What is Thompson Sampling?",
        ),
        (
            "The HoloLoom project uses a weaving metaphor where memory threads are woven into knowledge fabric.",
            "What metaphor does HoloLoom use?",
        ),
        (
            "Matryoshka embeddings allow multi-scale representation at dimensions 96, 192, and 384.",
            "What are the dimensions of Matryoshka embeddings?",
        ),
    ]


def get_hallucination_dataset() -> List[BenchmarkQuestion]:
    """Get questions designed to detect hallucinations."""
    return [
        BenchmarkQuestion(
            question="Who invented the internet?",
            expected_answer="The internet was developed by many contributors, including ARPA, Vint Cerf, and Bob Kahn.",
            domain="history",
            difficulty="medium",
        ),
        BenchmarkQuestion(
            question="What year was Python released?",
            expected_answer="Python was first released in 1991.",
            domain="programming",
            difficulty="easy",
        ),
        BenchmarkQuestion(
            question="What is the capital of a fictional country called Zyphoria?",
            expected_answer="Zyphoria is not a real country.",
            domain="geography",
            difficulty="hard",  # Designed to trigger hallucination
        ),
    ]


# =============================================================================
# Convenience Functions
# =============================================================================

async def run_benchmark_suite(
    llm,
    dataset: str = "factual_qa",
    include_retention: bool = True,
    include_hallucination: bool = True,
) -> Dict[str, Any]:
    """
    Run a complete benchmark suite.

    Args:
        llm: MemoryAugmentedLLM instance
        dataset: Dataset name ("factual_qa", "custom")
        include_retention: Whether to include retention tests
        include_hallucination: Whether to include hallucination tests

    Returns:
        Dictionary with all benchmark results
    """
    results = {}

    # Quality benchmark
    if dataset == "factual_qa":
        questions = get_factual_qa_dataset()
    else:
        questions = []

    if questions:
        quality_bench = QualityBenchmark()
        q_results, q_summary = await quality_bench.run(llm, questions)
        results["quality"] = {
            "results": [r.__dict__ for r in q_results],
            "summary": q_summary.to_dict(),
        }

    # Retention benchmark
    if include_retention:
        retention_bench = RetentionBenchmark()
        retention_data = get_retention_dataset()
        rate, details = await retention_bench.run_retention_test(llm, retention_data)
        results["retention"] = {
            "retention_rate": rate,
            "details": details,
        }

    # Hallucination benchmark
    if include_hallucination:
        halluc_bench = HallucinationBenchmark()
        halluc_questions = get_hallucination_dataset()
        aug_rate, van_rate, details = await halluc_bench.run_hallucination_test(
            llm, halluc_questions
        )
        results["hallucination"] = {
            "augmented_rate": aug_rate,
            "vanilla_rate": van_rate,
            "reduction": van_rate - aug_rate if van_rate > 0 else 0,
            "details": details,
        }

    return results


def generate_benchmark_report(results: Dict[str, Any]) -> str:
    """Generate a human-readable benchmark report."""
    lines = [
        "=" * 70,
        "MODEL EXTENSION SDK BENCHMARK REPORT",
        "=" * 70,
        "",
    ]

    # Quality section
    if "quality" in results:
        q = results["quality"]["summary"]
        lines.extend([
            "QUALITY BENCHMARK",
            "-" * 40,
            f"Questions Tested:     {q['total_questions']}",
            f"Average Quality:      {q['quality']['avg_score']:.2f}",
            f"Improvement:          {q['quality']['avg_improvement']:+.1%}",
            f"Augmented Latency:    {q['performance']['augmented_latency_ms']:.1f}ms",
            f"Vanilla Latency:      {q['performance']['vanilla_latency_ms']:.1f}ms",
            f"Memory Utilization:   {q['memory']['utilization_percent']:.1f}%",
            "",
        ])

    # Retention section
    if "retention" in results:
        r = results["retention"]
        lines.extend([
            "RETENTION BENCHMARK",
            "-" * 40,
            f"Retention Rate:       {r['retention_rate']:.1%}",
            f"Facts Tested:         {len(r['details'])}",
            "",
        ])

    # Hallucination section
    if "hallucination" in results:
        h = results["hallucination"]
        lines.extend([
            "HALLUCINATION BENCHMARK",
            "-" * 40,
            f"Augmented Rate:       {h['augmented_rate']:.1%}",
            f"Vanilla Rate:         {h['vanilla_rate']:.1%}",
            f"Reduction:            {h['reduction']:+.1%}",
            "",
        ])

    lines.append("=" * 70)
    return "\n".join(lines)


def save_benchmark_results(
    results: Dict[str, Any],
    filepath: str,
) -> None:
    """Save benchmark results to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_benchmark_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
