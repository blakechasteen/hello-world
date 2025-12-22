"""
Tests for Model Extension SDK Benchmark Utilities

Tests the benchmark system for comparing memory-augmented LLMs
against vanilla LLMs across quality, efficiency, retention,
hallucination, and domain-specific metrics.
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from HoloLoom.model_extension.benchmark import (
    BenchmarkType,
    BenchmarkQuestion,
    BenchmarkResult,
    BenchmarkSummary,
    BaseBenchmark,
    QualityBenchmark,
    EfficiencyBenchmark,
    RetentionBenchmark,
    HallucinationBenchmark,
    DomainBenchmark,
    get_factual_qa_dataset,
    get_retention_dataset,
    get_hallucination_dataset,
    run_benchmark_suite,
    generate_benchmark_report,
    save_benchmark_results,
    load_benchmark_results,
)


class TestBenchmarkType:
    """Tests for BenchmarkType enum."""

    def test_quality_type(self):
        """Test QUALITY benchmark type."""
        assert BenchmarkType.QUALITY.value == "quality"

    def test_efficiency_type(self):
        """Test EFFICIENCY benchmark type."""
        assert BenchmarkType.EFFICIENCY.value == "efficiency"

    def test_retention_type(self):
        """Test RETENTION benchmark type."""
        assert BenchmarkType.RETENTION.value == "retention"

    def test_hallucination_type(self):
        """Test HALLUCINATION benchmark type."""
        assert BenchmarkType.HALLUCINATION.value == "hallucination"

    def test_domain_type(self):
        """Test DOMAIN benchmark type."""
        assert BenchmarkType.DOMAIN.value == "domain"

    def test_all_types_exist(self):
        """Test all expected benchmark types exist."""
        expected = {"quality", "efficiency", "retention", "hallucination", "domain"}
        actual = {t.value for t in BenchmarkType}
        assert actual == expected


class TestBenchmarkQuestion:
    """Tests for BenchmarkQuestion dataclass."""

    def test_create_minimal(self):
        """Test creating question with minimal fields."""
        q = BenchmarkQuestion(question="What is Python?")

        assert q.question == "What is Python?"
        assert q.expected_answer is None
        assert q.domain == "general"
        assert q.difficulty == "medium"
        assert q.requires_memory is False
        assert q.metadata == {}

    def test_create_full(self):
        """Test creating question with all fields."""
        q = BenchmarkQuestion(
            question="Explain recursion",
            expected_answer="Recursion is when a function calls itself",
            domain="programming",
            difficulty="hard",
            requires_memory=True,
            metadata={"source": "test", "tags": ["cs"]},
        )

        assert q.question == "Explain recursion"
        assert q.expected_answer == "Recursion is when a function calls itself"
        assert q.domain == "programming"
        assert q.difficulty == "hard"
        assert q.requires_memory is True
        assert q.metadata["source"] == "test"

    def test_difficulty_levels(self):
        """Test different difficulty levels."""
        for difficulty in ["easy", "medium", "hard"]:
            q = BenchmarkQuestion(question="test", difficulty=difficulty)
            assert q.difficulty == difficulty


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_create_minimal(self):
        """Test creating result with minimal fields."""
        result = BenchmarkResult(
            question="What is AI?",
            augmented_answer="AI is artificial intelligence",
        )

        assert result.question == "What is AI?"
        assert result.augmented_answer == "AI is artificial intelligence"
        assert result.vanilla_answer is None
        assert result.augmented_confidence == 0.0
        assert result.vanilla_confidence == 0.0
        assert result.augmented_latency_ms == 0.0
        assert result.vanilla_latency_ms == 0.0
        assert result.augmented_sources == 0
        assert result.quality_score == 0.0
        assert result.improvement_score == 0.0
        assert result.used_memory is False

    def test_create_full(self):
        """Test creating result with all fields."""
        result = BenchmarkResult(
            question="What is ML?",
            augmented_answer="ML is machine learning",
            vanilla_answer="ML is a subset of AI",
            augmented_confidence=0.9,
            vanilla_confidence=0.7,
            augmented_latency_ms=150.5,
            vanilla_latency_ms=100.2,
            augmented_sources=3,
            quality_score=0.85,
            improvement_score=0.15,
            used_memory=True,
            metadata={"test": True},
        )

        assert result.augmented_confidence == 0.9
        assert result.vanilla_confidence == 0.7
        assert result.augmented_latency_ms == 150.5
        assert result.vanilla_latency_ms == 100.2
        assert result.augmented_sources == 3
        assert result.quality_score == 0.85
        assert result.improvement_score == 0.15
        assert result.used_memory is True
        assert result.metadata["test"] is True


class TestBenchmarkSummary:
    """Tests for BenchmarkSummary dataclass."""

    def test_create_summary(self):
        """Test creating benchmark summary."""
        summary = BenchmarkSummary(
            benchmark_type=BenchmarkType.QUALITY,
            total_questions=10,
            avg_quality_score=0.82,
            avg_improvement=0.15,
            avg_augmented_latency_ms=155.0,
            avg_vanilla_latency_ms=120.0,
            latency_overhead_percent=29.2,
            memory_utilization_percent=70.0,
            avg_sources_per_query=2.5,
            avg_confidence_augmented=0.88,
            avg_confidence_vanilla=0.75,
        )

        assert summary.benchmark_type == BenchmarkType.QUALITY
        assert summary.total_questions == 10
        assert summary.avg_quality_score == 0.82
        assert summary.avg_improvement == 0.15
        assert summary.latency_overhead_percent == 29.2

    def test_to_dict(self):
        """Test to_dict serialization."""
        summary = BenchmarkSummary(
            benchmark_type=BenchmarkType.QUALITY,
            total_questions=5,
            avg_quality_score=0.8,
            avg_improvement=0.1,
            avg_augmented_latency_ms=150.0,
            avg_vanilla_latency_ms=100.0,
            latency_overhead_percent=50.0,
            memory_utilization_percent=60.0,
            avg_sources_per_query=2.0,
            avg_confidence_augmented=0.85,
            avg_confidence_vanilla=0.7,
        )

        d = summary.to_dict()

        assert d["benchmark_type"] == "quality"
        assert d["total_questions"] == 5
        assert d["quality"]["avg_score"] == 0.8
        assert d["quality"]["avg_improvement"] == 0.1
        assert d["performance"]["augmented_latency_ms"] == 150.0
        assert d["performance"]["vanilla_latency_ms"] == 100.0
        assert d["performance"]["overhead_percent"] == 50.0
        assert d["memory"]["utilization_percent"] == 60.0
        assert d["memory"]["avg_sources_per_query"] == 2.0
        assert d["confidence"]["augmented"] == 0.85
        assert d["confidence"]["vanilla"] == 0.7
        assert "timestamp" in d

    def test_summary_string(self):
        """Test human-readable summary generation."""
        summary = BenchmarkSummary(
            benchmark_type=BenchmarkType.QUALITY,
            total_questions=10,
            avg_quality_score=0.85,
            avg_improvement=0.2,
            avg_augmented_latency_ms=150.0,
            avg_vanilla_latency_ms=100.0,
            latency_overhead_percent=50.0,
            memory_utilization_percent=80.0,
            avg_sources_per_query=3.0,
            avg_confidence_augmented=0.9,
            avg_confidence_vanilla=0.7,
        )

        text = summary.summary()

        assert "QUALITY" in text
        assert "10" in text  # Questions tested
        assert "0.85" in text  # Quality score
        assert "150" in text  # Latency
        assert "80" in text  # Memory utilization

    def test_timestamp_auto_populated(self):
        """Test timestamp is automatically populated."""
        before = datetime.now()
        summary = BenchmarkSummary(
            benchmark_type=BenchmarkType.QUALITY,
            total_questions=1,
            avg_quality_score=0.5,
            avg_improvement=0.0,
            avg_augmented_latency_ms=100.0,
            avg_vanilla_latency_ms=100.0,
            latency_overhead_percent=0.0,
            memory_utilization_percent=0.0,
            avg_sources_per_query=0.0,
            avg_confidence_augmented=0.5,
            avg_confidence_vanilla=0.5,
        )
        after = datetime.now()

        assert before <= summary.timestamp <= after


class TestBaseBenchmarkDefaultQualityScorer:
    """Tests for BaseBenchmark default quality scoring."""

    def test_empty_answer_scores_zero(self):
        """Test empty answer gets zero score."""
        benchmark = BaseBenchmark()
        score = benchmark._default_quality_scorer("What is X?", "", None)
        assert score == 0.0

    def test_short_answer_penalty(self):
        """Test very short answers get penalty."""
        benchmark = BaseBenchmark()
        score = benchmark._default_quality_scorer("What is X?", "Yes", None)
        assert score < 0.5

    def test_reasonable_length_bonus(self):
        """Test reasonable length answers get bonus."""
        benchmark = BaseBenchmark()
        # 15 words - reasonable length
        answer = " ".join(["word"] * 15)
        score = benchmark._default_quality_scorer("What is X?", answer, None)
        assert score >= 0.5

    def test_uncertainty_marker_penalty(self):
        """Test uncertainty markers reduce score."""
        benchmark = BaseBenchmark()
        answer = "I don't know what that is, but it might be related to AI."
        score = benchmark._default_quality_scorer("What is ML?", answer, None)
        # Should have penalty for "i don't know"
        assert score < 0.6

    def test_expected_answer_overlap_bonus(self):
        """Test matching expected answer gives bonus."""
        benchmark = BaseBenchmark()
        expected = "Machine learning is a subset of AI"
        answer = "Machine learning is definitely a subset of artificial intelligence and AI"
        score = benchmark._default_quality_scorer("What is ML?", answer, expected)
        # Good overlap with expected
        assert score > 0.5

    def test_score_clamped_0_to_1(self):
        """Test scores are clamped between 0 and 1."""
        benchmark = BaseBenchmark()

        # Test low score clamping
        score = benchmark._default_quality_scorer("Q", "x", None)
        assert 0.0 <= score <= 1.0

        # Test high score scenario
        expected = "word " * 20
        answer = expected
        score = benchmark._default_quality_scorer("Q", answer, expected)
        assert 0.0 <= score <= 1.0


class TestBaseBenchmarkComputeSummary:
    """Tests for BaseBenchmark summary computation."""

    def test_empty_results(self):
        """Test summary computation with empty results."""
        benchmark = BaseBenchmark()
        summary = benchmark._compute_summary([])

        assert summary.total_questions == 0
        assert summary.avg_quality_score == 0.0
        assert summary.avg_improvement == 0.0

    def test_single_result(self):
        """Test summary computation with single result."""
        benchmark = BaseBenchmark()
        results = [
            BenchmarkResult(
                question="Q1",
                augmented_answer="A1",
                vanilla_answer="V1",
                augmented_confidence=0.9,
                vanilla_confidence=0.7,
                augmented_latency_ms=150.0,
                vanilla_latency_ms=100.0,
                augmented_sources=2,
                quality_score=0.85,
                improvement_score=0.1,
                used_memory=True,
            )
        ]

        summary = benchmark._compute_summary(results)

        assert summary.total_questions == 1
        assert summary.avg_quality_score == 0.85
        assert summary.avg_improvement == 0.1
        assert summary.avg_augmented_latency_ms == 150.0
        assert summary.avg_vanilla_latency_ms == 100.0
        assert summary.memory_utilization_percent == 100.0
        assert summary.avg_sources_per_query == 2.0

    def test_multiple_results(self):
        """Test summary computation with multiple results."""
        benchmark = BaseBenchmark()
        results = [
            BenchmarkResult(
                question="Q1",
                augmented_answer="A1",
                vanilla_answer="V1",
                augmented_confidence=0.9,
                vanilla_confidence=0.7,
                augmented_latency_ms=100.0,
                vanilla_latency_ms=80.0,
                augmented_sources=2,
                quality_score=0.8,
                improvement_score=0.1,
                used_memory=True,
            ),
            BenchmarkResult(
                question="Q2",
                augmented_answer="A2",
                vanilla_answer="V2",
                augmented_confidence=0.8,
                vanilla_confidence=0.6,
                augmented_latency_ms=200.0,
                vanilla_latency_ms=120.0,
                augmented_sources=4,
                quality_score=0.9,
                improvement_score=0.2,
                used_memory=True,
            ),
        ]

        summary = benchmark._compute_summary(results)

        assert summary.total_questions == 2
        assert summary.avg_quality_score == pytest.approx(0.85)  # (0.8 + 0.9) / 2
        assert summary.avg_improvement == pytest.approx(0.15)  # (0.1 + 0.2) / 2
        assert summary.avg_augmented_latency_ms == 150.0  # (100 + 200) / 2
        assert summary.avg_sources_per_query == 3.0  # (2 + 4) / 2

    def test_latency_overhead_calculation(self):
        """Test latency overhead percentage calculation."""
        benchmark = BaseBenchmark()
        results = [
            BenchmarkResult(
                question="Q1",
                augmented_answer="A1",
                vanilla_answer="V1",
                augmented_latency_ms=150.0,
                vanilla_latency_ms=100.0,
            )
        ]

        summary = benchmark._compute_summary(results)

        # Overhead = ((150 - 100) / 100) * 100 = 50%
        assert summary.latency_overhead_percent == 50.0


class TestQualityBenchmark:
    """Tests for QualityBenchmark class."""

    def test_benchmark_type(self):
        """Test QualityBenchmark has correct type."""
        benchmark = QualityBenchmark()
        assert benchmark.benchmark_type == BenchmarkType.QUALITY

    def test_custom_quality_scorer(self):
        """Test using custom quality scorer."""
        def custom_scorer(q, a, e):
            return 1.0 if "correct" in a.lower() else 0.0

        benchmark = QualityBenchmark(quality_scorer=custom_scorer)

        score = benchmark.quality_scorer("Q", "This is correct", None)
        assert score == 1.0

        score = benchmark.quality_scorer("Q", "This is wrong", None)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_run_with_mock_llm(self):
        """Test running benchmark with mocked LLM."""
        # Create mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.answer = "This is a test answer with enough words."
        mock_response.confidence = 0.85
        mock_response.sources = ["source1", "source2"]
        mock_llm.query = AsyncMock(return_value=mock_response)

        benchmark = QualityBenchmark()
        questions = [
            BenchmarkQuestion(question="What is test?", expected_answer="A test"),
        ]

        results, summary = await benchmark.run(
            mock_llm, questions, compare_vanilla=False
        )

        assert len(results) == 1
        assert results[0].augmented_answer == "This is a test answer with enough words."
        assert results[0].augmented_confidence == 0.85
        assert results[0].augmented_sources == 2
        assert summary.total_questions == 1


class TestEfficiencyBenchmark:
    """Tests for EfficiencyBenchmark class."""

    def test_benchmark_type(self):
        """Test EfficiencyBenchmark has correct type."""
        benchmark = EfficiencyBenchmark()
        assert benchmark.benchmark_type == BenchmarkType.EFFICIENCY

    @pytest.mark.asyncio
    async def test_run_with_token_tracking(self):
        """Test efficiency benchmark with token tracking."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.answer = "Efficient answer"
        mock_response.confidence = 0.9
        mock_response.sources = []
        mock_llm.query = AsyncMock(return_value=mock_response)

        benchmark = EfficiencyBenchmark()
        questions = [BenchmarkQuestion(question="Test?")]

        results, summary, token_stats = await benchmark.run_with_token_tracking(
            mock_llm, questions
        )

        assert len(results) == 1
        assert summary.total_questions == 1
        assert isinstance(token_stats, dict)
        assert "total_prompt_tokens" in token_stats


class TestRetentionBenchmark:
    """Tests for RetentionBenchmark class."""

    def test_benchmark_type(self):
        """Test RetentionBenchmark has correct type."""
        benchmark = RetentionBenchmark()
        assert benchmark.benchmark_type == BenchmarkType.RETENTION

    @pytest.mark.asyncio
    async def test_run_retention_test(self):
        """Test retention benchmark."""
        mock_llm = MagicMock()

        # Mock learn
        mock_llm.learn = AsyncMock()

        # Mock query to return answer containing taught facts
        mock_response = MagicMock()
        mock_response.answer = "Thompson Sampling is a Bayesian approach"
        mock_response.confidence = 0.9
        mock_response.sources = ["learned_fact"]
        mock_llm.query = AsyncMock(return_value=mock_response)

        benchmark = RetentionBenchmark()
        teach_facts = [
            ("Thompson Sampling is a Bayesian approach", "What is Thompson Sampling?"),
        ]

        rate, details = await benchmark.run_retention_test(
            mock_llm, teach_facts, delay_seconds=0.0
        )

        # Should have learned the fact
        mock_llm.learn.assert_called_once()

        # Should have high retention rate due to word overlap
        assert rate >= 0.0
        assert len(details) == 1
        assert details[0]["question"] == "What is Thompson Sampling?"

    @pytest.mark.asyncio
    async def test_retention_with_no_overlap(self):
        """Test retention when answer has no overlap with fact."""
        mock_llm = MagicMock()
        mock_llm.learn = AsyncMock()

        mock_response = MagicMock()
        mock_response.answer = "I have no idea what you're asking about."
        mock_response.confidence = 0.2
        mock_response.sources = []
        mock_llm.query = AsyncMock(return_value=mock_response)

        benchmark = RetentionBenchmark()
        teach_facts = [
            ("Specific technical fact XYZ", "What is XYZ?"),
        ]

        rate, details = await benchmark.run_retention_test(mock_llm, teach_facts)

        # Low retention due to no overlap and no sources
        assert rate == 0.0
        assert details[0]["retained"] is False


class TestHallucinationBenchmark:
    """Tests for HallucinationBenchmark class."""

    def test_benchmark_type(self):
        """Test HallucinationBenchmark has correct type."""
        benchmark = HallucinationBenchmark()
        assert benchmark.benchmark_type == BenchmarkType.HALLUCINATION

    def test_default_hallucination_detector_no_markers(self):
        """Test hallucination detector with clean answer."""
        benchmark = HallucinationBenchmark()
        score = benchmark._default_hallucination_detector(
            "What is Python?",
            "Python is a programming language used for various applications."
        )
        # Should have low hallucination score
        assert score < 0.5

    def test_default_hallucination_detector_overconfident(self):
        """Test hallucination detector with overconfident markers."""
        benchmark = HallucinationBenchmark()
        score = benchmark._default_hallucination_detector(
            "Is Python fast?",
            "Python is definitely always the fastest language, 100% certain."
        )
        # Should have higher hallucination score
        assert score > 0.1

    def test_default_hallucination_detector_specific_numbers(self):
        """Test hallucination detector with specific numbers."""
        benchmark = HallucinationBenchmark()
        score = benchmark._default_hallucination_detector(
            "How many users?",
            "There are exactly 12345678 users worldwide as of today."
        )
        # Should flag specific numbers
        assert score > 0.0

    @pytest.mark.asyncio
    async def test_run_hallucination_test(self):
        """Test running hallucination benchmark."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.answer = "A reasonable answer about Python."
        mock_response.confidence = 0.8
        mock_response.sources = []
        mock_llm.query = AsyncMock(return_value=mock_response)

        benchmark = HallucinationBenchmark()
        questions = [
            BenchmarkQuestion(question="What is Python?"),
        ]

        aug_rate, van_rate, details = await benchmark.run_hallucination_test(
            mock_llm, questions
        )

        assert 0.0 <= aug_rate <= 1.0
        assert 0.0 <= van_rate <= 1.0
        assert len(details) == 1
        assert "augmented_hallucination_score" in details[0]
        assert "vanilla_hallucination_score" in details[0]

    @pytest.mark.asyncio
    async def test_custom_hallucination_detector(self):
        """Test using custom hallucination detector."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.answer = "hallucinated content"
        mock_response.confidence = 0.5
        mock_response.sources = []
        mock_llm.query = AsyncMock(return_value=mock_response)

        def custom_detector(q, a):
            return 1.0 if "hallucinated" in a else 0.0

        benchmark = HallucinationBenchmark()
        questions = [BenchmarkQuestion(question="Test?")]

        aug_rate, van_rate, details = await benchmark.run_hallucination_test(
            mock_llm, questions, hallucination_detector=custom_detector
        )

        # Custom detector should flag as hallucination
        assert aug_rate == 1.0


class TestDomainBenchmark:
    """Tests for DomainBenchmark class."""

    def test_benchmark_type(self):
        """Test DomainBenchmark has correct type."""
        benchmark = DomainBenchmark(
            domain="medical",
            domain_knowledge=["Medical fact 1"],
        )
        assert benchmark.benchmark_type == BenchmarkType.DOMAIN

    def test_domain_initialization(self):
        """Test domain benchmark initialization."""
        knowledge = ["Fact 1", "Fact 2", "Fact 3"]
        benchmark = DomainBenchmark(
            domain="legal",
            domain_knowledge=knowledge,
        )

        assert benchmark.domain == "legal"
        assert benchmark.domain_knowledge == knowledge

    @pytest.mark.asyncio
    async def test_run_domain_test(self):
        """Test running domain-specific benchmark."""
        mock_llm = MagicMock()
        mock_llm.learn = AsyncMock()

        mock_response = MagicMock()
        mock_response.answer = "Domain specific answer about medical topics."
        mock_response.confidence = 0.85
        mock_response.sources = []
        mock_llm.query = AsyncMock(return_value=mock_response)

        benchmark = DomainBenchmark(
            domain="medical",
            domain_knowledge=["Medical knowledge fact 1"],
        )

        questions = [
            BenchmarkQuestion(
                question="Medical question?",
                expected_answer="Medical answer",
                domain="medical",
            ),
        ]

        before_score, after_score, summary = await benchmark.run_domain_test(
            mock_llm, questions
        )

        # learn should have been called for domain knowledge
        mock_llm.learn.assert_called_once()

        assert 0.0 <= before_score <= 1.0
        assert 0.0 <= after_score <= 1.0
        assert summary.total_questions == 1


class TestStandardDatasets:
    """Tests for standard benchmark datasets."""

    def test_factual_qa_dataset(self):
        """Test factual Q&A dataset."""
        dataset = get_factual_qa_dataset()

        assert len(dataset) >= 5
        assert all(isinstance(q, BenchmarkQuestion) for q in dataset)
        assert all(q.question for q in dataset)
        assert all(q.expected_answer for q in dataset)

    def test_factual_qa_diverse_domains(self):
        """Test factual dataset has diverse domains."""
        dataset = get_factual_qa_dataset()
        domains = {q.domain for q in dataset}

        assert len(domains) >= 3  # Should have multiple domains

    def test_factual_qa_diverse_difficulties(self):
        """Test factual dataset has diverse difficulties."""
        dataset = get_factual_qa_dataset()
        difficulties = {q.difficulty for q in dataset}

        assert "easy" in difficulties
        assert "medium" in difficulties

    def test_retention_dataset(self):
        """Test retention dataset."""
        dataset = get_retention_dataset()

        assert len(dataset) >= 3
        assert all(isinstance(item, tuple) for item in dataset)
        assert all(len(item) == 2 for item in dataset)
        assert all(item[0] and item[1] for item in dataset)  # Non-empty strings

    def test_hallucination_dataset(self):
        """Test hallucination dataset."""
        dataset = get_hallucination_dataset()

        assert len(dataset) >= 3
        assert all(isinstance(q, BenchmarkQuestion) for q in dataset)

        # Should include a question designed to trigger hallucination
        hard_questions = [q for q in dataset if q.difficulty == "hard"]
        assert len(hard_questions) >= 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_run_benchmark_suite_with_factual_qa(self):
        """Test running full benchmark suite."""
        mock_llm = MagicMock()
        mock_llm.learn = AsyncMock()

        mock_response = MagicMock()
        mock_response.answer = "Test answer with reasonable content."
        mock_response.confidence = 0.8
        mock_response.sources = ["source"]
        mock_llm.query = AsyncMock(return_value=mock_response)

        results = await run_benchmark_suite(
            mock_llm,
            dataset="factual_qa",
            include_retention=True,
            include_hallucination=True,
        )

        assert "quality" in results
        assert "retention" in results
        assert "hallucination" in results
        assert "summary" in results["quality"]
        assert "retention_rate" in results["retention"]
        assert "augmented_rate" in results["hallucination"]

    @pytest.mark.asyncio
    async def test_run_benchmark_suite_quality_only(self):
        """Test running benchmark suite without retention/hallucination."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.answer = "Answer"
        mock_response.confidence = 0.7
        mock_response.sources = []
        mock_llm.query = AsyncMock(return_value=mock_response)

        results = await run_benchmark_suite(
            mock_llm,
            dataset="factual_qa",
            include_retention=False,
            include_hallucination=False,
        )

        assert "quality" in results
        assert "retention" not in results
        assert "hallucination" not in results

    @pytest.mark.asyncio
    async def test_run_benchmark_suite_empty_dataset(self):
        """Test running benchmark suite with unknown dataset."""
        mock_llm = MagicMock()
        mock_llm.learn = AsyncMock()
        mock_response = MagicMock()
        mock_response.answer = "Answer"
        mock_response.confidence = 0.7
        mock_response.sources = []
        mock_llm.query = AsyncMock(return_value=mock_response)

        results = await run_benchmark_suite(
            mock_llm,
            dataset="unknown_dataset",
            include_retention=True,
            include_hallucination=True,
        )

        # Should skip quality but still run retention/hallucination
        assert "quality" not in results
        assert "retention" in results
        assert "hallucination" in results


class TestGenerateBenchmarkReport:
    """Tests for generate_benchmark_report function."""

    def test_report_with_all_sections(self):
        """Test report generation with all benchmark sections."""
        results = {
            "quality": {
                "summary": {
                    "total_questions": 10,
                    "quality": {"avg_score": 0.85, "avg_improvement": 0.15},
                    "performance": {
                        "augmented_latency_ms": 150.0,
                        "vanilla_latency_ms": 100.0,
                    },
                    "memory": {"utilization_percent": 70.0},
                },
            },
            "retention": {
                "retention_rate": 0.9,
                "details": [{"fact": "f1"}, {"fact": "f2"}],
            },
            "hallucination": {
                "augmented_rate": 0.1,
                "vanilla_rate": 0.3,
                "reduction": 0.2,
            },
        }

        report = generate_benchmark_report(results)

        assert "BENCHMARK REPORT" in report
        assert "QUALITY BENCHMARK" in report
        assert "RETENTION BENCHMARK" in report
        assert "HALLUCINATION BENCHMARK" in report
        assert "0.85" in report  # Quality score
        assert "90" in report  # Retention rate (0.9 = 90%)

    def test_report_with_quality_only(self):
        """Test report generation with quality only."""
        results = {
            "quality": {
                "summary": {
                    "total_questions": 5,
                    "quality": {"avg_score": 0.8, "avg_improvement": 0.1},
                    "performance": {
                        "augmented_latency_ms": 120.0,
                        "vanilla_latency_ms": 100.0,
                    },
                    "memory": {"utilization_percent": 60.0},
                },
            },
        }

        report = generate_benchmark_report(results)

        assert "QUALITY BENCHMARK" in report
        assert "RETENTION BENCHMARK" not in report
        assert "HALLUCINATION BENCHMARK" not in report


class TestSaveAndLoadResults:
    """Tests for save/load benchmark results functions."""

    def test_save_and_load_results(self):
        """Test saving and loading benchmark results."""
        results = {
            "quality": {
                "summary": {
                    "total_questions": 5,
                    "avg_quality": 0.85,
                },
                "results": [
                    {"question": "Q1", "answer": "A1"},
                ],
            },
            "timestamp": datetime.now().isoformat(),
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            save_benchmark_results(results, filepath)

            # Verify file exists
            assert os.path.exists(filepath)

            # Load and verify
            loaded = load_benchmark_results(filepath)
            assert loaded["quality"]["summary"]["total_questions"] == 5
            assert loaded["quality"]["summary"]["avg_quality"] == 0.85

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_save_results_with_complex_data(self):
        """Test saving results with complex nested data."""
        results = {
            "quality": {
                "results": [
                    {
                        "question": "Q1",
                        "metadata": {"nested": {"deep": True}},
                    }
                ],
            },
            "retention": {
                "details": [
                    {"fact": "F1", "retained": True, "score": 0.9},
                ],
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            save_benchmark_results(results, filepath)
            loaded = load_benchmark_results(filepath)

            assert loaded["quality"]["results"][0]["metadata"]["nested"]["deep"] is True
            assert loaded["retention"]["details"][0]["retained"] is True

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestBenchmarkIntegration:
    """Integration tests for benchmark system."""

    @pytest.mark.asyncio
    async def test_full_benchmark_flow(self):
        """Test full benchmark flow from question to report."""
        # Create comprehensive mock
        mock_llm = MagicMock()
        mock_llm.learn = AsyncMock()

        mock_response = MagicMock()
        mock_response.answer = "Machine learning is a subset of AI that enables learning from data."
        mock_response.confidence = 0.88
        mock_response.sources = ["doc1", "doc2"]
        mock_llm.query = AsyncMock(return_value=mock_response)

        # Run quality benchmark
        quality_bench = QualityBenchmark()
        questions = get_factual_qa_dataset()[:2]  # Just first 2

        results, summary = await quality_bench.run(
            mock_llm, questions, compare_vanilla=True
        )

        # Verify results
        assert len(results) == 2
        assert all(r.augmented_answer for r in results)
        assert summary.total_questions == 2
        assert summary.avg_quality_score > 0

        # Generate report
        full_results = {
            "quality": {
                "results": [r.__dict__ for r in results],
                "summary": summary.to_dict(),
            }
        }
        report = generate_benchmark_report(full_results)
        assert "QUALITY BENCHMARK" in report

    @pytest.mark.asyncio
    async def test_benchmark_error_handling(self):
        """Test benchmark handles LLM errors gracefully."""
        mock_llm = MagicMock()
        mock_llm.query = AsyncMock(side_effect=Exception("LLM Error"))

        benchmark = QualityBenchmark()
        questions = [BenchmarkQuestion(question="Test?")]

        results, summary = await benchmark.run(
            mock_llm, questions, compare_vanilla=False
        )

        # Should handle error and continue
        assert len(results) == 1
        assert "Error" in results[0].augmented_answer
        assert results[0].augmented_confidence == 0.0

    @pytest.mark.asyncio
    async def test_multiple_benchmarks_same_llm(self):
        """Test running multiple benchmarks on same LLM."""
        mock_llm = MagicMock()
        mock_llm.learn = AsyncMock()

        mock_response = MagicMock()
        mock_response.answer = "Consistent answer"
        mock_response.confidence = 0.8
        mock_response.sources = []
        mock_llm.query = AsyncMock(return_value=mock_response)

        # Run multiple benchmarks
        quality_bench = QualityBenchmark()
        efficiency_bench = EfficiencyBenchmark()

        questions = [BenchmarkQuestion(question="Q?")]

        q_results, q_summary = await quality_bench.run(mock_llm, questions)
        e_results, e_summary = await efficiency_bench.run(mock_llm, questions)

        assert q_summary.benchmark_type == BenchmarkType.QUALITY
        assert e_summary.benchmark_type == BenchmarkType.EFFICIENCY
