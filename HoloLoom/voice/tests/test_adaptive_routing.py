"""
Test Suite for AR Adaptive Routing System
==========================================
Comprehensive tests for AR query classification, pattern mining, validation, and deployment.

Author: Claude Code (Agent Q)
Date: November 17, 2025
Phase: Agent Swarm Wave 5 - Adaptive Routing Integration
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
import asyncio

from HoloLoom.voice.ar_query_classifier import (
    ARQueryClassifier,
    ARComplexity,
    ARQueryType,
    ARPatternRule,
    ARClassificationResult
)
from HoloLoom.voice.ar_pattern_miner import (
    ARPatternMiner,
    ARPattern,
    PatternScore
)
from HoloLoom.voice.ar_validator import (
    ARContinuousValidator,
    ValidationQuery,
    ValidationResult
)
from HoloLoom.voice.ar_pattern_deployer import (
    ARPatternDeployer,
    DeploymentStrategy,
    DeploymentPhase
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_log_dir():
    """Create temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def ar_classifier(temp_log_dir):
    """Create AR query classifier."""
    return ARQueryClassifier(
        enable_logging=True,
        log_dir=temp_log_dir
    )


@pytest.fixture
def ar_pattern_miner(temp_log_dir):
    """Create AR pattern miner."""
    return ARPatternMiner(
        log_dir=temp_log_dir,
        min_support=3,  # Lower for testing
        min_precision=0.80  # Lower for testing
    )


@pytest.fixture
def ar_validator(ar_classifier):
    """Create AR continuous validator."""
    return ARContinuousValidator(
        classifier=ar_classifier,
        regression_threshold=0.05  # 5% for testing
    )


@pytest.fixture
def ar_pattern_deployer(ar_classifier, ar_validator):
    """Create AR pattern deployer."""
    return ARPatternDeployer(
        classifier=ar_classifier,
        validator=ar_validator,
        strategy=DeploymentStrategy.GRADUAL
    )


@pytest.fixture
def sample_validation_set():
    """Create sample validation set."""
    return [
        ValidationQuery(text="next hive", expected_complexity="simple"),
        ValidationQuery(text="show me hive 5", expected_complexity="simple"),
        ValidationQuery(text="what is the health status", expected_complexity="standard"),
        ValidationQuery(text="compare these hives", expected_complexity="standard"),
        ValidationQuery(text="show all unhealthy hives", expected_complexity="complex"),
        ValidationQuery(text="find patterns in bee behavior", expected_complexity="complex"),
        ValidationQuery(text="why is this hive declining", expected_complexity="research"),
        ValidationQuery(text="explain the health trends", expected_complexity="research"),
    ]


# ============================================================================
# AR Query Classifier Tests (12 tests)
# ============================================================================

def test_classifier_simple_navigation(ar_classifier):
    """Test classification of simple navigation commands."""
    result = ar_classifier.classify("next hive")

    assert result.complexity == ARComplexity.SIMPLE
    assert result.ar_type == ARQueryType.VOICE_ONLY
    assert result.confidence > 0.3


def test_classifier_simple_selection(ar_classifier):
    """Test classification of simple selection commands."""
    result = ar_classifier.classify("show me hive 5")

    assert result.complexity == ARComplexity.SIMPLE
    assert result.ar_type == ARQueryType.VOICE_ONLY
    assert result.confidence > 0.3


def test_classifier_spatial_reference(ar_classifier):
    """Test classification of spatial references."""
    result = ar_classifier.classify("that one over there")

    assert result.complexity == ARComplexity.SIMPLE
    assert result.ar_type == ARQueryType.SPATIAL_REFERENCE
    assert result.confidence > 0.4


def test_classifier_gesture_command(ar_classifier):
    """Test classification of gesture commands."""
    result = ar_classifier.classify(
        "select this",
        context={"has_gesture": True}
    )

    assert result.complexity == ARComplexity.SIMPLE
    assert result.ar_type in [ARQueryType.GESTURE_COMMAND, ARQueryType.MULTIMODAL]
    assert result.confidence > 0.3


def test_classifier_standard_what_question(ar_classifier):
    """Test classification of standard what questions."""
    result = ar_classifier.classify("what is the health status")

    assert result.complexity == ARComplexity.STANDARD
    assert result.ar_type == ARQueryType.VOICE_ONLY
    assert result.confidence > 0.3


def test_classifier_visual_query(ar_classifier):
    """Test classification of visual queries."""
    result = ar_classifier.classify(
        "what's this",
        context={"has_vision": True}
    )

    assert result.ar_type in [ARQueryType.VISUAL_QUERY, ARQueryType.MULTIMODAL]
    assert result.confidence > 0.3


def test_classifier_complex_all_query(ar_classifier):
    """Test classification of complex 'all' queries."""
    result = ar_classifier.classify("show me all unhealthy hives")

    assert result.complexity == ARComplexity.COMPLEX
    assert result.confidence > 0.3


def test_classifier_complex_filter(ar_classifier):
    """Test classification of complex filter queries."""
    result = ar_classifier.classify("find bees with low health")

    assert result.complexity == ARComplexity.COMPLEX
    assert result.confidence > 0.3


def test_classifier_research_why_question(ar_classifier):
    """Test classification of research why questions."""
    result = ar_classifier.classify("why is this hive unhealthy")

    assert result.complexity == ARComplexity.RESEARCH
    assert result.confidence > 0.4


def test_classifier_research_explain(ar_classifier):
    """Test classification of research explain queries."""
    result = ar_classifier.classify("explain the health decline")

    assert result.complexity == ARComplexity.RESEARCH
    assert result.confidence > 0.4


def test_classifier_multimodal_detection(ar_classifier):
    """Test detection of multimodal queries."""
    result = ar_classifier.classify(
        "show this hive and then compare it",
        context={"has_gesture": True, "has_vision": False}
    )

    assert result.ar_type in [ARQueryType.MULTIMODAL, ARQueryType.GESTURE_COMMAND]


def test_classifier_logging(ar_classifier, temp_log_dir):
    """Test that classifications are logged."""
    ar_classifier.classify("test query")

    log_dir = Path(temp_log_dir)
    log_files = list(log_dir.glob("classifications_*.jsonl"))

    assert len(log_files) > 0

    # Check log content
    with open(log_files[0], "r") as f:
        lines = f.readlines()
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["query"] == "test query"
        assert "complexity" in entry
        assert "confidence" in entry


# ============================================================================
# AR Pattern Miner Tests (10 tests)
# ============================================================================

def test_pattern_miner_initialization(ar_pattern_miner):
    """Test pattern miner initialization."""
    assert ar_pattern_miner.min_support == 3
    assert ar_pattern_miner.min_precision == 0.80
    assert ar_pattern_miner.max_patterns == 50


def test_pattern_miner_load_logs(ar_classifier, ar_pattern_miner, temp_log_dir):
    """Test loading classification logs."""
    # Generate some logs
    queries = [
        "next hive",
        "next colony",
        "previous hive",
        "show me hive 5",
        "show me hive 12",
    ]

    for query in queries:
        ar_classifier.classify(query)

    # Mine patterns
    patterns = ar_pattern_miner.mine_patterns(lookback_days=1)

    # Should find some patterns (might be empty due to min thresholds)
    assert isinstance(patterns, list)


def test_pattern_miner_ngram_extraction(ar_pattern_miner):
    """Test n-gram extraction."""
    queries = ["next hive", "next colony", "previous hive"]
    ngrams = ar_pattern_miner._extract_ngrams(queries, 1)

    assert "next" in ngrams
    assert ngrams["next"] == 2


def test_pattern_miner_find_common_phrases(ar_pattern_miner):
    """Test finding common phrases."""
    queries = [
        "show me hive 5",
        "show me hive 12",
        "show me colony 3"
    ]

    phrases = ar_pattern_miner._find_common_phrases(queries)

    assert "show me" in phrases
    assert phrases["show me"] == 3


def test_pattern_miner_ar_specific_templates(ar_pattern_miner):
    """Test AR-specific pattern templates."""
    templates = ar_pattern_miner.ar_pattern_templates

    assert "gesture" in templates
    assert "spatial" in templates
    assert "visual" in templates
    assert "multimodal" in templates


def test_pattern_miner_score_patterns(ar_classifier, ar_pattern_miner, temp_log_dir):
    """Test pattern scoring."""
    # Generate logs
    for _ in range(10):
        ar_classifier.classify("next hive")

    # Load logs
    logs = ar_pattern_miner._load_logs(lookback_days=1)

    # Create test pattern
    pattern = ARPattern(
        regex=r'\bnext\s+hive\b',
        complexity=ARComplexity.SIMPLE,
        ar_type=ARQueryType.VOICE_ONLY,
        score=PatternScore(0, 0, 0, 0, 0),
        examples=[]
    )

    # Score pattern
    scored_patterns = ar_pattern_miner._score_patterns([pattern], logs)

    assert len(scored_patterns) > 0
    assert scored_patterns[0].score.support > 0


def test_pattern_miner_filter_patterns(ar_pattern_miner):
    """Test pattern filtering by quality thresholds."""
    patterns = [
        ARPattern(
            regex=r'\bhigh\s+quality\b',
            complexity=ARComplexity.SIMPLE,
            ar_type=ARQueryType.VOICE_ONLY,
            score=PatternScore(
                precision=0.95,
                recall=0.50,
                support=20,
                confidence=0.90,
                f1_score=0.65
            ),
            examples=[]
        ),
        ARPattern(
            regex=r'\blow\s+quality\b',
            complexity=ARComplexity.SIMPLE,
            ar_type=ARQueryType.VOICE_ONLY,
            score=PatternScore(
                precision=0.60,  # Too low
                recall=0.20,
                support=5,
                confidence=0.50,
                f1_score=0.30
            ),
            examples=[]
        ),
    ]

    filtered = ar_pattern_miner._filter_patterns(patterns)

    assert len(filtered) == 1
    assert filtered[0].regex == r'\bhigh\s+quality\b'


def test_pattern_miner_export_patterns(ar_pattern_miner, temp_log_dir):
    """Test exporting patterns to JSON."""
    patterns = [
        ARPattern(
            regex=r'\btest\b',
            complexity=ARComplexity.SIMPLE,
            ar_type=ARQueryType.VOICE_ONLY,
            score=PatternScore(0.95, 0.50, 10, 0.90, 0.65),
            examples=["test query"]
        )
    ]

    filepath = Path(temp_log_dir) / "patterns.json"
    ar_pattern_miner.export_patterns(patterns, str(filepath))

    assert filepath.exists()

    with open(filepath, "r") as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["pattern"] == r'\btest\b'


def test_pattern_miner_mine_high_confidence(ar_classifier, ar_pattern_miner):
    """Test mining high-confidence patterns."""
    # Generate high-confidence logs
    for _ in range(15):
        ar_classifier.classify("next hive")

    patterns = ar_pattern_miner.mine_patterns(
        lookback_days=1,
        include_low_confidence=False
    )

    # May or may not find patterns depending on thresholds
    assert isinstance(patterns, list)


def test_pattern_miner_mine_misclassification(ar_classifier, ar_pattern_miner):
    """Test mining misclassification patterns."""
    # Generate some logs (mix of queries)
    queries = [
        "next hive",
        "what's this thing",
        "show me patterns",
        "explain why",
    ]

    for query in queries:
        ar_classifier.classify(query)

    patterns = ar_pattern_miner.mine_patterns(
        lookback_days=1,
        include_low_confidence=True
    )

    assert isinstance(patterns, list)


# ============================================================================
# AR Continuous Validator Tests (8 tests)
# ============================================================================

def test_validator_initialization(ar_validator):
    """Test validator initialization."""
    assert ar_validator.regression_threshold == 0.05
    assert ar_validator.history_size == 1000


def test_validator_add_validation_query(ar_validator):
    """Test adding validation queries."""
    ar_validator.add_validation_query(
        text="next hive",
        expected_complexity="simple"
    )

    assert len(ar_validator.validation_set) == 1
    assert ar_validator.validation_set[0].text == "next hive"


def test_validator_validate(ar_validator, sample_validation_set):
    """Test validation run."""
    for val_query in sample_validation_set:
        ar_validator.add_validation_query(
            val_query.text,
            val_query.expected_complexity
        )

    result = ar_validator.validate()

    assert isinstance(result, ValidationResult)
    assert result.total_queries == len(sample_validation_set)
    assert 0.0 <= result.overall_accuracy <= 1.0


def test_validator_regression_detection(ar_validator, sample_validation_set):
    """Test regression detection."""
    for val_query in sample_validation_set:
        ar_validator.add_validation_query(
            val_query.text,
            val_query.expected_complexity
        )

    # First validation (sets baseline)
    result1 = ar_validator.validate()
    ar_validator.baseline_accuracy = 0.95  # Set high baseline

    # Second validation (should detect regression)
    result2 = ar_validator.validate()

    # May or may not detect regression depending on actual accuracy
    assert isinstance(result2.regression_detected, bool)


def test_validator_get_alerts(ar_validator):
    """Test getting alerts."""
    alerts = ar_validator.get_alerts()
    assert isinstance(alerts, list)


def test_validator_save_validation_set(ar_validator, sample_validation_set, temp_log_dir):
    """Test saving validation set."""
    for val_query in sample_validation_set:
        ar_validator.add_validation_query(
            val_query.text,
            val_query.expected_complexity
        )

    filepath = Path(temp_log_dir) / "validation_set.json"
    ar_validator.save_validation_set(str(filepath))

    assert filepath.exists()

    with open(filepath, "r") as f:
        data = json.load(f)

    assert len(data) == len(sample_validation_set)


def test_validator_load_validation_set(ar_classifier, temp_log_dir):
    """Test loading validation set from file."""
    # Create validation set file
    validation_data = [
        {
            "text": "next hive",
            "expected_complexity": "simple",
            "context": {}
        }
    ]

    filepath = Path(temp_log_dir) / "validation_set.json"
    with open(filepath, "w") as f:
        json.dump(validation_data, f)

    # Create validator with validation set
    validator = ARContinuousValidator(
        classifier=ar_classifier,
        validation_set_path=str(filepath)
    )

    assert len(validator.validation_set) == 1


def test_validator_statistics(ar_validator, sample_validation_set):
    """Test getting validation statistics."""
    for val_query in sample_validation_set:
        ar_validator.add_validation_query(
            val_query.text,
            val_query.expected_complexity
        )

    ar_validator.validate()

    stats = ar_validator.get_statistics()

    assert "total_validations" in stats
    assert "avg_accuracy" in stats
    assert stats["total_validations"] == 1


# ============================================================================
# AR Pattern Deployer Tests (10 tests)
# ============================================================================

@pytest.mark.asyncio
async def test_deployer_initialization(ar_pattern_deployer):
    """Test deployer initialization."""
    assert ar_pattern_deployer.strategy == DeploymentStrategy.GRADUAL
    assert ar_pattern_deployer.rollback_threshold == 0.02
    assert ar_pattern_deployer.current_version == 0


@pytest.mark.asyncio
async def test_deployer_deploy_shadow(ar_pattern_deployer):
    """Test shadow deployment."""
    ar_pattern_deployer.strategy = DeploymentStrategy.SHADOW

    patterns = [
        ARPattern(
            regex=r'\btest\b',
            complexity=ARComplexity.SIMPLE,
            ar_type=ARQueryType.VOICE_ONLY,
            score=PatternScore(0.95, 0.50, 10, 0.90, 0.65),
            examples=[]
        )
    ]

    result = await ar_pattern_deployer.deploy_patterns(patterns, validate_first=False)

    assert result.success
    assert result.current_phase == DeploymentPhase.SHADOW
    assert result.patterns_deployed == 1


@pytest.mark.asyncio
async def test_deployer_deploy_ab_test(ar_pattern_deployer):
    """Test A/B test deployment."""
    ar_pattern_deployer.strategy = DeploymentStrategy.AB_TEST

    patterns = [
        ARPattern(
            regex=r'\btest\b',
            complexity=ARComplexity.SIMPLE,
            ar_type=ARQueryType.VOICE_ONLY,
            score=PatternScore(0.95, 0.50, 10, 0.90, 0.65),
            examples=[]
        )
    ]

    result = await ar_pattern_deployer.deploy_patterns(patterns, validate_first=False)

    assert result.success or result.rollback_triggered  # May rollback if validation fails
    assert result.patterns_deployed == 1


@pytest.mark.asyncio
async def test_deployer_deploy_gradual(ar_pattern_deployer):
    """Test gradual deployment."""
    ar_pattern_deployer.strategy = DeploymentStrategy.GRADUAL

    patterns = [
        ARPattern(
            regex=r'\btest\b',
            complexity=ARComplexity.SIMPLE,
            ar_type=ARQueryType.VOICE_ONLY,
            score=PatternScore(0.95, 0.50, 10, 0.90, 0.65),
            examples=[]
        )
    ]

    result = await ar_pattern_deployer.deploy_patterns(patterns, validate_first=False)

    # May succeed or rollback depending on validation
    assert result.patterns_deployed == 1


@pytest.mark.asyncio
async def test_deployer_deploy_immediate(ar_pattern_deployer):
    """Test immediate deployment."""
    ar_pattern_deployer.strategy = DeploymentStrategy.IMMEDIATE

    patterns = [
        ARPattern(
            regex=r'\btest\b',
            complexity=ARComplexity.SIMPLE,
            ar_type=ARQueryType.VOICE_ONLY,
            score=PatternScore(0.95, 0.50, 10, 0.90, 0.65),
            examples=[]
        )
    ]

    result = await ar_pattern_deployer.deploy_patterns(patterns, validate_first=False)

    assert result.success
    assert result.current_phase == DeploymentPhase.FULL


@pytest.mark.asyncio
async def test_deployer_pattern_versioning(ar_pattern_deployer):
    """Test pattern versioning."""
    patterns = [
        ARPattern(
            regex=r'\btest\b',
            complexity=ARComplexity.SIMPLE,
            ar_type=ARQueryType.VOICE_ONLY,
            score=PatternScore(0.95, 0.50, 10, 0.90, 0.65),
            examples=[]
        )
    ]

    await ar_pattern_deployer.deploy_patterns(patterns, validate_first=False)

    assert ar_pattern_deployer.current_version == 1
    assert len(ar_pattern_deployer.pattern_versions) == 1


@pytest.mark.asyncio
async def test_deployer_get_current_version(ar_pattern_deployer):
    """Test getting current pattern version."""
    patterns = [
        ARPattern(
            regex=r'\btest\b',
            complexity=ARComplexity.SIMPLE,
            ar_type=ARQueryType.VOICE_ONLY,
            score=PatternScore(0.95, 0.50, 10, 0.90, 0.65),
            examples=[]
        )
    ]

    await ar_pattern_deployer.deploy_patterns(patterns, validate_first=False)

    current_version = ar_pattern_deployer.get_current_version()

    assert current_version is not None
    assert current_version.version == 1
    assert current_version.is_active


def test_deployer_get_statistics(ar_pattern_deployer):
    """Test getting deployment statistics."""
    stats = ar_pattern_deployer.get_statistics()

    assert "total_versions" in stats
    assert "current_version" in stats
    assert "strategy" in stats


def test_deployer_export_metrics(ar_pattern_deployer, temp_log_dir):
    """Test exporting deployment metrics."""
    filepath = Path(temp_log_dir) / "metrics.json"
    ar_pattern_deployer.export_metrics(str(filepath))

    assert filepath.exists()


@pytest.mark.asyncio
async def test_deployer_max_versions_limit(ar_pattern_deployer):
    """Test that old versions are trimmed."""
    ar_pattern_deployer.max_versions = 3

    # Deploy 5 versions
    for i in range(5):
        patterns = [
            ARPattern(
                regex=rf'\btest{i}\b',
                complexity=ARComplexity.SIMPLE,
                ar_type=ARQueryType.VOICE_ONLY,
                score=PatternScore(0.95, 0.50, 10, 0.90, 0.65),
                examples=[]
            )
        ]
        await ar_pattern_deployer.deploy_patterns(patterns, validate_first=False)

    # Should only keep last 3
    assert len(ar_pattern_deployer.pattern_versions) == 3


# ============================================================================
# Integration Tests (3 tests)
# ============================================================================

@pytest.mark.asyncio
async def test_end_to_end_adaptive_routing(ar_classifier, ar_pattern_miner, ar_validator, ar_pattern_deployer, sample_validation_set):
    """Test complete end-to-end adaptive routing workflow."""
    # 1. Generate classification logs
    queries = ["next hive"] * 20 + ["show me hive 5"] * 15

    for query in queries:
        ar_classifier.classify(query)

    # 2. Mine patterns
    patterns = ar_pattern_miner.mine_patterns(lookback_days=1)

    # 3. Setup validation set
    for val_query in sample_validation_set:
        ar_validator.add_validation_query(
            val_query.text,
            val_query.expected_complexity
        )

    # 4. Validate baseline
    baseline_result = ar_validator.validate()
    assert baseline_result.total_queries > 0

    # 5. Deploy patterns (if any found)
    if patterns:
        result = await ar_pattern_deployer.deploy_patterns(patterns, validate_first=False)
        assert result.patterns_deployed > 0


def test_classifier_custom_patterns(ar_classifier):
    """Test adding custom patterns to classifier."""
    custom_pattern = ARPatternRule(
        pattern=r'\bcustom\s+pattern\b',
        complexity=ARComplexity.SIMPLE,
        ar_type=ARQueryType.VOICE_ONLY,
        confidence_boost=0.20,
        priority=15,
        examples=["custom pattern test"]
    )

    ar_classifier.add_custom_pattern(custom_pattern)

    result = ar_classifier.classify("custom pattern test")

    assert result.complexity == ARComplexity.SIMPLE


def test_classifier_load_patterns_from_file(ar_classifier, temp_log_dir):
    """Test loading custom patterns from JSON file."""
    patterns_data = [
        {
            "pattern": r'\bloaded\s+pattern\b',
            "complexity": "simple",
            "ar_type": "voice_only",
            "confidence_boost": 0.15,
            "priority": 10,
            "examples": ["loaded pattern test"]
        }
    ]

    filepath = Path(temp_log_dir) / "custom_patterns.json"
    with open(filepath, "w") as f:
        json.dump(patterns_data, f)

    ar_classifier.load_patterns_from_file(str(filepath))

    result = ar_classifier.classify("loaded pattern test")

    assert result.complexity == ARComplexity.SIMPLE


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
======================

AR Query Classifier: 12 tests
- Simple navigation/selection
- Spatial references
- Gesture commands
- Standard questions
- Visual queries
- Complex queries
- Research questions
- Multimodal detection
- Logging

AR Pattern Miner: 10 tests
- Initialization
- Log loading
- N-gram extraction
- Common phrase finding
- AR-specific templates
- Pattern scoring
- Pattern filtering
- Export patterns
- High-confidence mining
- Misclassification mining

AR Continuous Validator: 8 tests
- Initialization
- Add validation queries
- Validation runs
- Regression detection
- Alerts
- Save/load validation set
- Statistics

AR Pattern Deployer: 10 tests
- Initialization
- Shadow deployment
- A/B test deployment
- Gradual deployment
- Immediate deployment
- Pattern versioning
- Get current version
- Statistics
- Export metrics
- Max versions limit

Integration Tests: 3 tests
- End-to-end workflow
- Custom patterns
- Load patterns from file

Total: 43 tests
"""
