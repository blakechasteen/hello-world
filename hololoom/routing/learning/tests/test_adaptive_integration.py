"""
Adaptive Learning Integration Tests
====================================
End-to-end tests for Phase 3 adaptive learning system.

Tests:
1. Full integration test (all 4 components)
2. Background learning loop
3. Pattern mining → validation → deployment → reporting pipeline
4. Regression detection and rollback
5. Performance metrics and reporting

Author: Claude Code
Date: November 13, 2025
"""

import pytest
import asyncio
import json
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from hololoom.routing.learning import (
    PatternMiner,
    ContinuousValidator,
    AdaptiveUpdater,
    PerformanceReporter,
    Pattern,
    PatternScore,
    create_validation_set,
    DeploymentStrategy
)


class MockClassifier:
    """Mock classifier for testing."""
    def __init__(self, accuracy=1.0):
        self.accuracy = accuracy
        self.patterns = []

    def classify(self, query_text: str):
        """Mock classification."""
        from dataclasses import dataclass

        @dataclass
        class MockClassification:
            complexity: str
            confidence: float

        class ComplexityValue:
            def __init__(self, value):
                self.value = value

        # Simple mock logic
        if len(query_text.split()) <= 3:
            complexity = "trivial"
        elif "?" in query_text:
            complexity = "simple"
        elif "explain" in query_text.lower() or "describe" in query_text.lower():
            complexity = "complex"
        else:
            complexity = "research"

        return MockClassification(
            complexity=ComplexityValue(complexity),
            confidence=self.accuracy
        )


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def validation_set(temp_dir):
    """Create test validation set."""
    queries = [
        ("hi", "trivial"),
        ("hello", "trivial"),
        ("what is Python?", "simple"),
        ("what are transformers?", "simple"),
        ("explain neural networks", "complex"),
        ("describe attention mechanism", "complex"),
        ("analyze tradeoffs of RL algorithms", "research"),
        ("compare supervised vs unsupervised learning", "research"),
    ]

    val_file = temp_dir / "validation_set.json"
    create_validation_set(queries, str(val_file))
    return val_file


@pytest.fixture
def classifier():
    """Create mock classifier."""
    return MockClassifier(accuracy=1.0)


@pytest.fixture
def classification_log(temp_dir):
    """Create mock classification log."""
    log_file = temp_dir / "classifications.jsonl"

    # Generate sample classification logs
    logs = [
        {"timestamp": "2025-11-13T00:00:00", "query_text": "hi", "complexity": "trivial", "confidence": 1.0},
        {"timestamp": "2025-11-13T00:01:00", "query_text": "hello", "complexity": "trivial", "confidence": 1.0},
        {"timestamp": "2025-11-13T00:02:00", "query_text": "what is Python?", "complexity": "simple", "confidence": 0.95},
        {"timestamp": "2025-11-13T00:03:00", "query_text": "what are transformers?", "complexity": "simple", "confidence": 0.95},
        {"timestamp": "2025-11-13T00:04:00", "query_text": "explain neural networks", "complexity": "complex", "confidence": 0.90},
        {"timestamp": "2025-11-13T00:05:00", "query_text": "describe attention mechanism", "complexity": "complex", "confidence": 0.90},
        {"timestamp": "2025-11-13T00:06:00", "query_text": "analyze RL tradeoffs", "complexity": "research", "confidence": 0.85},
        {"timestamp": "2025-11-13T00:07:00", "query_text": "compare learning paradigms", "complexity": "research", "confidence": 0.85},
    ]

    with open(log_file, 'w', encoding='utf-8') as f:
        for log in logs:
            f.write(json.dumps(log) + '\n')

    return log_file


class TestPatternMiner:
    """Test PatternMiner component."""

    def test_pattern_miner_initialization(self, classification_log):
        """Test PatternMiner initializes correctly."""
        miner = PatternMiner(
            stats_path=str(classification_log),
            min_support=2,
            min_precision=0.90,
            min_recall=0.30
        )

        assert miner.min_support == 2
        assert miner.min_precision == 0.90
        assert miner.min_recall == 0.30

    def test_pattern_mining_finds_patterns(self, classification_log):
        """Test pattern mining discovers patterns."""
        miner = PatternMiner(
            stats_path=str(classification_log),
            min_support=1,
            min_precision=0.80,
            min_recall=0.10
        )

        patterns = miner.mine_patterns(days_lookback=7)

        # Should find some patterns (exact count depends on implementation)
        assert isinstance(patterns, list)
        # All patterns should have required attributes
        for pattern in patterns:
            assert hasattr(pattern, 'regex')
            assert hasattr(pattern, 'complexity')
            assert hasattr(pattern, 'score')


class TestContinuousValidator:
    """Test ContinuousValidator component."""

    @pytest.mark.asyncio
    async def test_validator_initialization(self, classifier, validation_set):
        """Test ContinuousValidator initializes correctly."""
        validator = ContinuousValidator(
            classifier=classifier,
            validation_set_path=str(validation_set),
            regression_threshold=0.02
        )

        assert validator.regression_threshold == 0.02
        assert len(validator.validation_set) > 0

    @pytest.mark.asyncio
    async def test_hourly_validation(self, classifier, validation_set):
        """Test hourly validation runs correctly."""
        validator = ContinuousValidator(
            classifier=classifier,
            validation_set_path=str(validation_set),
            regression_threshold=0.02,
            baseline_accuracy=1.0
        )

        result = await validator.validate_hourly(sample_size=4)

        assert result.overall_accuracy >= 0.0
        assert result.overall_accuracy <= 1.0
        assert result.total_queries > 0
        assert len(validator.validation_history) == 1

    @pytest.mark.asyncio
    async def test_regression_detection(self, validation_set):
        """Test regression detection triggers alerts."""
        # Create classifier with low accuracy to trigger regression
        bad_classifier = MockClassifier(accuracy=0.50)

        validator = ContinuousValidator(
            classifier=bad_classifier,
            validation_set_path=str(validation_set),
            regression_threshold=0.02,
            baseline_accuracy=1.0
        )

        result = await validator.validate_hourly(sample_size=8)

        # Should detect regression (50% vs 100% baseline)
        assert result.regression_detected == True
        assert len(validator.alerts) > 0


class TestAdaptiveUpdater:
    """Test AdaptiveUpdater component."""

    @pytest.mark.asyncio
    async def test_updater_initialization(self, classifier, validation_set):
        """Test AdaptiveUpdater initializes correctly."""
        validator = ContinuousValidator(
            classifier=classifier,
            validation_set_path=str(validation_set)
        )

        updater = AdaptiveUpdater(
            classifier=classifier,
            validator=validator,
            strategy=DeploymentStrategy.SHADOW
        )

        assert updater.strategy == DeploymentStrategy.SHADOW
        assert updater.traffic_split == 0.0  # Shadow mode = 0% traffic

    @pytest.mark.asyncio
    async def test_shadow_deployment(self, classifier, validation_set):
        """Test shadow mode deployment."""
        validator = ContinuousValidator(
            classifier=classifier,
            validation_set_path=str(validation_set)
        )

        updater = AdaptiveUpdater(
            classifier=classifier,
            validator=validator,
            strategy=DeploymentStrategy.SHADOW
        )

        test_patterns = [
            Pattern(
                regex=r"^hi$",
                complexity="trivial",
                score=PatternScore(
                    precision=1.0,
                    recall=0.95,
                    support=100,
                    confidence=0.98,
                    f1_score=0.97
                ),
                examples=["hi"]
            )
        ]

        result = await updater.deploy_patterns(test_patterns)

        assert result.success == True
        assert result.patterns_deployed == 1
        # Shadow mode should have 0% traffic
        assert updater.traffic_split == 0.0

    @pytest.mark.asyncio
    async def test_automatic_rollback(self, validation_set):
        """Test automatic rollback on regression."""
        # Create bad classifier that will trigger rollback
        bad_classifier = MockClassifier(accuracy=0.50)

        validator = ContinuousValidator(
            classifier=bad_classifier,
            validation_set_path=str(validation_set),
            baseline_accuracy=1.0
        )

        updater = AdaptiveUpdater(
            classifier=bad_classifier,
            validator=validator,
            strategy=DeploymentStrategy.AB_TEST,
            regression_threshold=0.02
        )

        test_patterns = [
            Pattern(
                regex=r"^test$",
                complexity="trivial",
                score=PatternScore(precision=1.0, recall=0.9, support=50, confidence=0.95, f1_score=0.94),
                examples=["test"]
            )
        ]

        result = await updater.deploy_patterns(test_patterns)

        # Should trigger rollback due to low accuracy
        assert result.rollback_triggered == True


class TestPerformanceReporter:
    """Test PerformanceReporter component."""

    @pytest.mark.asyncio
    async def test_reporter_initialization(self, classifier, validation_set, temp_dir):
        """Test PerformanceReporter initializes correctly."""
        validator = ContinuousValidator(
            classifier=classifier,
            validation_set_path=str(validation_set)
        )

        updater = AdaptiveUpdater(classifier=classifier, validator=validator)

        reporter = PerformanceReporter(
            validator=validator,
            updater=updater,
            output_dir=str(temp_dir / "reports")
        )

        assert reporter.validator == validator
        assert reporter.updater == updater

    @pytest.mark.asyncio
    async def test_daily_report_generation(self, classifier, validation_set, temp_dir):
        """Test daily report generation."""
        validator = ContinuousValidator(
            classifier=classifier,
            validation_set_path=str(validation_set),
            baseline_accuracy=1.0
        )

        # Run some validations to populate history
        await validator.validate_hourly(sample_size=4)
        await validator.validate_hourly(sample_size=4)

        updater = AdaptiveUpdater(classifier=classifier, validator=validator)

        reporter = PerformanceReporter(
            validator=validator,
            updater=updater,
            output_dir=str(temp_dir / "reports")
        )

        daily_report = reporter.generate_daily_report()

        assert daily_report.date is not None
        assert daily_report.queries_classified >= 0
        assert 0.0 <= daily_report.overall_accuracy <= 1.0

    @pytest.mark.asyncio
    async def test_prometheus_metrics_export(self, classifier, validation_set, temp_dir):
        """Test Prometheus metrics export."""
        validator = ContinuousValidator(
            classifier=classifier,
            validation_set_path=str(validation_set)
        )

        await validator.validate_hourly(sample_size=4)

        updater = AdaptiveUpdater(classifier=classifier, validator=validator)

        reporter = PerformanceReporter(
            validator=validator,
            updater=updater,
            output_dir=str(temp_dir / "reports")
        )

        metrics = reporter.export_prometheus_metrics()

        # Should contain Prometheus format metrics
        assert "moonshot_accuracy" in metrics
        assert "moonshot_queries_total" in metrics
        assert "# TYPE" in metrics  # Prometheus metadata


class TestEndToEndIntegration:
    """Test complete end-to-end adaptive learning pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, classifier, validation_set, classification_log, temp_dir):
        """Test complete adaptive learning pipeline."""
        # 1. Pattern Mining
        miner = PatternMiner(
            stats_path=str(classification_log),
            min_support=1,
            min_precision=0.80
        )

        patterns = miner.mine_patterns(days_lookback=7)
        assert isinstance(patterns, list)

        # 2. Continuous Validation
        validator = ContinuousValidator(
            classifier=classifier,
            validation_set_path=str(validation_set),
            regression_threshold=0.02,
            baseline_accuracy=1.0
        )

        validation_result = await validator.validate_hourly(sample_size=4)
        assert validation_result.overall_accuracy > 0.0

        # 3. Adaptive Updater
        updater = AdaptiveUpdater(
            classifier=classifier,
            validator=validator,
            strategy=DeploymentStrategy.SHADOW
        )

        if patterns:
            deployment_result = await updater.deploy_patterns(patterns[:3])
            assert deployment_result.success

        # 4. Performance Reporter
        reporter = PerformanceReporter(
            validator=validator,
            updater=updater,
            pattern_miner=miner,
            output_dir=str(temp_dir / "reports")
        )

        daily_report = reporter.generate_daily_report()
        assert daily_report.overall_accuracy >= 0.0

        weekly_report = reporter.generate_weekly_report()
        assert len(weekly_report.recommendations) > 0

        metrics = reporter.export_prometheus_metrics()
        assert len(metrics) > 0

    @pytest.mark.asyncio
    async def test_metrics_consistency(self, classifier, validation_set, temp_dir):
        """Test that metrics remain consistent across components."""
        validator = ContinuousValidator(
            classifier=classifier,
            validation_set_path=str(validation_set),
            baseline_accuracy=1.0
        )

        # Run multiple validations
        results = []
        for _ in range(3):
            result = await validator.validate_hourly(sample_size=4)
            results.append(result)

        # Check history matches
        assert len(validator.validation_history) == 3

        # Check all results have consistent structure
        for result in results:
            assert hasattr(result, 'overall_accuracy')
            assert hasattr(result, 'by_complexity')
            assert hasattr(result, 'total_queries')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
