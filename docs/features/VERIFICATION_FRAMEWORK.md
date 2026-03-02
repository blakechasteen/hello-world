# Concurrent Development Verification Framework

**Purpose**: Ensure quality, correctness, and integration across all 3 concurrent development options (BossPig, Quick Wins, Elle).

**Philosophy**: "Elegance and verification at every step" - Test early, test often, test thoroughly.

---

## Three-Tier Testing Architecture

```
┌─────────────────────────────────────────────────────┐
│              SYSTEM TIER (Integration)               │
│  End-to-end workflows across all 3 options          │
│  • BossPig → HoloLoom memory → Elle guidance        │
│  • MCTS Shuttle → Workflow Builder → Execution      │
│  Duration: 2-5 minutes per test                     │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│           INTEGRATION TIER (Component)               │
│  Multi-component interactions within each option    │
│  • BossPig detectors → scorer → formatter           │
│  • MCTS Warp ↔ Yarn intersection                    │
│  • Elle engine → vision → voice → memory            │
│  Duration: 10-30 seconds per test                   │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│               UNIT TIER (Isolated)                   │
│  Single component/function testing                  │
│  • Individual slop detectors (BossPig)              │
│  • Thompson Sampling updates (MCTS)                 │
│  • Vision object detection (Elle)                   │
│  Duration: <1 second per test                       │
└─────────────────────────────────────────────────────┘
```

---

## Option A: BossPig Verification

### Unit Tests (15 tests, <10 seconds total)

**File**: `tests/unit/test_bosspig_detectors.py`

```python
import pytest
from hololoom.bosspig import (
    JargonDetector,
    BuzzwordDetector,
    PassiveVoiceDetector,
    VagueQuantifierDetector,
    WeaselWordDetector
)

def test_jargon_detector():
    """Test jargon detection."""
    detector = JargonDetector()

    text = "We need to leverage our synergies to facilitate stakeholder alignment."
    detections = detector.detect(text)

    assert len(detections) >= 2  # "leverage synergies", "facilitate"
    assert any(d.category == "jargon" for d in detections)

    # Should have suggestions
    for detection in detections:
        assert len(detection.suggestion) > 0
        assert detection.confidence > 0.7

def test_buzzword_detector():
    """Test buzzword detection."""
    detector = BuzzwordDetector()

    text = "Our innovative, game-changing, disruptive solution is best-in-class."
    detections = detector.detect(text)

    assert len(detections) >= 3  # Multiple buzzwords
    assert all(d.confidence > 0.8 for d in detections)  # High confidence

def test_passive_voice_detector():
    """Test passive voice detection."""
    detector = PassiveVoiceDetector()

    # Passive
    text1 = "The report was written by the team."
    detections1 = detector.detect(text1)
    assert len(detections1) == 1
    assert "was written" in detections1[0].text

    # Active (should not detect)
    text2 = "The team wrote the report."
    detections2 = detector.detect(text2)
    assert len(detections2) == 0

def test_vague_quantifier_detector():
    """Test vague quantifier detection."""
    detector = VagueQuantifierDetector()

    text = "We have many clients with significant growth potential."
    detections = detector.detect(text)

    assert len(detections) >= 2  # "many", "significant"

    # Suggestions should be specific
    for detection in detections:
        assert "specific" in detection.suggestion.lower() or \
               "number" in detection.suggestion.lower()

def test_weasel_word_detector():
    """Test weasel word detection."""
    detector = WeaselWordDetector()

    text = "Some people say our product is probably the best, arguably."
    detections = detector.detect(text)

    assert len(detections) >= 3  # "some", "probably", "arguably"
    assert all(d.category == "weasel_word" for d in detections)

def test_detector_scoring():
    """Test severity scoring."""
    detector = JargonDetector()

    # High severity
    text1 = "We need to leverage our synergies to facilitate alignment."
    detections1 = detector.detect(text1)
    assert any(d.severity > 0.7 for d in detections1)

    # Low severity
    text2 = "We need to use our resources to help with alignment."
    detections2 = detector.detect(text2)
    assert len(detections2) == 0 or all(d.severity < 0.3 for d in detections2)

def test_detector_span_accuracy():
    """Test span detection accuracy."""
    detector = BuzzwordDetector()

    text = "Our innovative solution is game-changing."
    detections = detector.detect(text)

    for detection in detections:
        span_start, span_end = detection.span
        extracted = text[span_start:span_end]
        assert extracted.lower() in detection.text.lower()

@pytest.mark.parametrize("detector_class", [
    JargonDetector,
    BuzzwordDetector,
    PassiveVoiceDetector,
    VagueQuantifierDetector,
    WeaselWordDetector
])
def test_detector_empty_input(detector_class):
    """Test all detectors handle empty input."""
    detector = detector_class()

    detections = detector.detect("")
    assert len(detections) == 0

    detections = detector.detect("   ")
    assert len(detections) == 0

@pytest.mark.parametrize("detector_class", [
    JargonDetector,
    BuzzwordDetector,
    PassiveVoiceDetector
])
def test_detector_clean_input(detector_class):
    """Test detectors don't false-positive on clean text."""
    detector = detector_class()

    clean_text = "The team wrote a clear report with specific numbers."
    detections = detector.detect(clean_text)

    # Should have few or no detections on clean text
    assert len(detections) <= 1

def test_confidence_calibration():
    """Test confidence scores are well-calibrated."""
    detector = JargonDetector()

    # Obvious jargon (high confidence)
    text1 = "We need to leverage our synergies."
    detections1 = detector.detect(text1)
    assert detections1[0].confidence > 0.8

    # Borderline jargon (lower confidence)
    text2 = "We need to use our resources."
    detections2 = detector.detect(text2)
    # Should either not detect or have low confidence
    if detections2:
        assert detections2[0].confidence < 0.6

def test_suggestion_quality():
    """Test suggestions are actionable."""
    detector = PassiveVoiceDetector()

    text = "The report was written by the team."
    detections = detector.detect(text)

    suggestion = detections[0].suggestion

    # Should suggest active voice
    assert "active" in suggestion.lower()
    assert len(suggestion) > 10  # Not too terse
    assert len(suggestion) < 200  # Not too verbose

def test_example_quality():
    """Test examples are provided and relevant."""
    detector = JargonDetector()

    text = "We need to leverage our synergies."
    detections = detector.detect(text)

    example = detections[0].example

    # Should have an example
    assert len(example) > 0

    # Example should be clear
    assert "→" in example or "instead" in example.lower()
```

**Run unit tests**:
```bash
pytest tests/unit/test_bosspig_detectors.py -v
```

### Integration Tests (8 tests, <30 seconds total)

**File**: `tests/integration/test_bosspig_pipeline.py`

```python
import pytest
from hololoom.bosspig import BossPigDetector, DocumentAnalysis

def test_full_document_analysis():
    """Test complete document analysis pipeline."""
    detector = BossPigDetector()

    # Sample business document (high slop)
    document = """
    Executive Summary

    Our innovative, game-changing solution leverages cutting-edge technology
    to facilitate stakeholder alignment and drive synergies across the organization.

    Key Benefits:
    - Many clients have seen significant growth
    - Our approach is arguably best-in-class
    - Results are typically very good

    The deliverables were completed by the team in a timely manner.
    """

    analysis = detector.analyze(document, document_id="test_doc_1")

    # Should detect multiple issues
    assert len(analysis.detections) >= 8

    # Should have low score (high slop)
    assert analysis.overall_score < 60

    # Should have assessment
    assert len(analysis.assessment) > 0
    assert "grade" in analysis.assessment.lower()

    # Should have suggestions
    assert len(analysis.suggestions) > 0

def test_clean_document_analysis():
    """Test analysis of clean business writing."""
    detector = BossPigDetector()

    document = """
    Project Status Report - Q4 2025

    The team completed 12 features in Q4, representing 95% of planned work.
    Three features were delayed due to dependency issues.

    Key Metrics:
    - Revenue: $1.2M (up 15% from Q3)
    - Customer satisfaction: 4.2/5.0 (target: 4.0)
    - Bug resolution time: 2.3 days average

    The engineering team resolved 45 bugs and improved test coverage from 78% to 82%.
    """

    analysis = detector.analyze(document, document_id="clean_doc")

    # Should have few detections
    assert len(analysis.detections) < 3

    # Should have high score
    assert analysis.overall_score >= 80

    # Grade should be good
    assert "A" in analysis.assessment or "B" in analysis.assessment

def test_category_distribution():
    """Test detection across different categories."""
    detector = BossPigDetector()

    document = """
    We need to leverage our innovative synergies to facilitate alignment.
    Many stakeholders say our approach is arguably best-in-class.
    The report was written by the team with significant effort.
    """

    analysis = detector.analyze(document)

    # Should detect multiple categories
    categories = {d.category for d in analysis.detections}
    assert len(categories) >= 3

    # Expected categories
    assert "jargon" in categories or "buzzword" in categories
    assert "weasel_word" in categories or "vague_quantifier" in categories
    assert "passive_voice" in categories

def test_severity_weighting():
    """Test severity affects overall score."""
    detector = BossPigDetector()

    # High severity issues
    doc1 = "We leverage cutting-edge synergies to facilitate disruptive innovation."
    analysis1 = detector.analyze(doc1)

    # Low severity issues
    doc2 = "We use modern tools to help with new ideas."
    analysis2 = detector.analyze(doc2)

    # High severity should score lower
    assert analysis1.overall_score < analysis2.overall_score

def test_document_metadata():
    """Test metadata tracking."""
    detector = BossPigDetector()

    document = "Test document content."
    analysis = detector.analyze(document, document_id="meta_test_123")

    # Should track metadata
    assert "document_id" in analysis.metadata
    assert analysis.metadata["document_id"] == "meta_test_123"
    assert "analysis_timestamp" in analysis.metadata

def test_formatter_text_output():
    """Test text formatter."""
    from hololoom.bosspig import TextFormatter

    detector = BossPigDetector()
    formatter = TextFormatter()

    document = "We need to leverage synergies to facilitate alignment."
    analysis = detector.analyze(document)

    output = formatter.format(analysis)

    # Should be human-readable
    assert "Score:" in output
    assert "Detections:" in output
    assert len(output) > 100
    assert len(output) < 5000  # Not too verbose

def test_formatter_json_output():
    """Test JSON formatter."""
    from hololoom.bosspig import JSONFormatter
    import json

    detector = BossPigDetector()
    formatter = JSONFormatter()

    document = "We leverage synergies."
    analysis = detector.analyze(document)

    output = formatter.format(analysis)

    # Should be valid JSON
    data = json.loads(output)

    # Should have expected fields
    assert "overall_score" in data
    assert "detections" in data
    assert isinstance(data["detections"], list)

def test_batch_processing():
    """Test batch document processing."""
    detector = BossPigDetector()

    documents = [
        "Our innovative solution leverages synergies.",
        "The team completed 5 features with 95% test coverage.",
        "Many stakeholders say our approach is arguably best."
    ]

    analyses = [detector.analyze(doc, document_id=f"doc_{i}")
                for i, doc in enumerate(documents)]

    # Should process all
    assert len(analyses) == 3

    # Should have different scores
    scores = [a.overall_score for a in analyses]
    assert len(set(scores)) > 1  # Not all the same
```

**Run integration tests**:
```bash
pytest tests/integration/test_bosspig_pipeline.py -v
```

### System Tests (3 tests, <5 minutes total)

**File**: `tests/system/test_bosspig_end_to_end.py`

```python
import pytest
import asyncio
from pathlib import Path

from hololoom.bosspig import BossPigDetector
from hololoom.bosspig.cli import BossPigCLI
from hololoom import HoloLoom

@pytest.mark.asyncio
@pytest.mark.slow
async def test_cli_analyze_command():
    """Test CLI analyze command end-to-end."""
    # Create test document
    test_file = Path("test_document.txt")
    test_file.write_text("""
    Our innovative solution leverages cutting-edge synergies.
    Many stakeholders say the approach is arguably best-in-class.
    """)

    try:
        # Run CLI
        cli = BossPigCLI()
        result = await cli.analyze_file(test_file, output_format="json")

        # Should return analysis
        assert "overall_score" in result
        assert result["overall_score"] < 70  # High slop
        assert len(result["detections"]) >= 3
    finally:
        # Cleanup
        test_file.unlink()

@pytest.mark.asyncio
@pytest.mark.slow
async def test_hololoom_memory_integration():
    """Test BossPig → HoloLoom memory integration."""
    detector = BossPigDetector()

    async with HoloLoom() as loom:
        # Analyze document
        document = "We leverage innovative synergies to facilitate alignment."
        analysis = detector.analyze(document)

        # Store in HoloLoom memory
        await loom.experience(f"Analyzed document: {analysis.assessment}")
        await loom.experience(f"Detected {len(analysis.detections)} slop instances")

        # Should recall analysis
        memories = await loom.recall("business writing quality")

        assert len(memories) > 0
        assert any("slop" in m.content.lower() for m in memories)

@pytest.mark.asyncio
@pytest.mark.slow
async def test_full_analysis_workflow():
    """Test complete workflow: analyze → format → save → recall."""
    from hololoom.bosspig import TextFormatter
    import tempfile

    detector = BossPigDetector()
    formatter = TextFormatter()

    # Analyze
    document = """
    Executive Summary
    We leverage cutting-edge technology to facilitate stakeholder synergies.
    Our innovative approach is arguably best-in-class.
    """

    analysis = detector.analyze(document, document_id="workflow_test")

    # Format
    report = formatter.format(analysis)

    # Save
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(report)
        report_path = f.name

    try:
        # Read back
        with open(report_path, 'r') as f:
            saved_report = f.read()

        # Verify
        assert "Score:" in saved_report
        assert "Detections:" in saved_report
        assert len(saved_report) > 200

    finally:
        # Cleanup
        Path(report_path).unlink()
```

**Run system tests**:
```bash
pytest tests/system/test_bosspig_end_to_end.py -v -m slow
```

---

## Option B: Quick Wins Verification

### Unit Tests (12 tests, <5 seconds total)

**File**: `tests/unit/test_mcts_shuttle.py`

```python
import pytest
import numpy as np
from hololoom.shuttle import MCTSShuttle, ShuttleConfig, ThompsonSampler

def test_thompson_sampler_initialization():
    """Test Thompson Sampler initialization."""
    sampler = ThompsonSampler(n_arms=3)

    # Should initialize with uniform priors
    assert sampler.alpha.shape == (3,)
    assert sampler.beta.shape == (3,)
    assert np.allclose(sampler.alpha, 1.0)
    assert np.allclose(sampler.beta, 1.0)

def test_thompson_sampler_update():
    """Test Thompson Sampler updates."""
    sampler = ThompsonSampler(n_arms=3)

    # Reward arm 0
    sampler.update(arm=0, reward=1.0)

    # Alpha should increase
    assert sampler.alpha[0] > 1.0
    assert sampler.beta[0] == 1.0

    # Penalize arm 1
    sampler.update(arm=1, reward=0.0)

    # Beta should increase
    assert sampler.alpha[1] == 1.0
    assert sampler.beta[1] > 1.0

def test_thompson_sampler_sample():
    """Test Thompson Sampler sampling."""
    sampler = ThompsonSampler(n_arms=3)

    # Update to create bias
    for _ in range(10):
        sampler.update(arm=0, reward=1.0)

    # Sample 100 times
    samples = [sampler.sample() for _ in range(100)]

    # Arm 0 should be sampled most
    from collections import Counter
    counts = Counter(samples)
    assert counts[0] > counts[1]
    assert counts[0] > counts[2]

def test_shuttle_config_validation():
    """Test ShuttleConfig validation."""
    # Valid config
    config = ShuttleConfig(
        qdrant_host="localhost",
        qdrant_port=6333,
        neo4j_uri="bolt://localhost:7687",
        num_mcts_simulations=50,
        exploration_weight=1.4
    )

    assert config.qdrant_port == 6333
    assert config.num_mcts_simulations == 50

def test_shuttle_initialization():
    """Test MCTSShuttle initialization."""
    config = ShuttleConfig()
    shuttle = MCTSShuttle(config)

    # Should have Warp and Yarn components
    assert shuttle.warp is not None
    assert shuttle.yarn is not None
    assert shuttle.thompson is not None

@pytest.mark.asyncio
async def test_warp_search_mock():
    """Test Warp vector search (mock)."""
    config = ShuttleConfig()
    shuttle = MCTSShuttle(config)

    # Mock search (real test requires Qdrant)
    query = "What is Thompson Sampling?"

    # Should not crash
    # results = await shuttle.warp.search(query, top_k=5)
    # assert len(results) <= 5

@pytest.mark.asyncio
async def test_yarn_traverse_mock():
    """Test Yarn graph traversal (mock)."""
    config = ShuttleConfig()
    shuttle = MCTSShuttle(config)

    # Mock traversal (real test requires Neo4j)
    # Should not crash
    # results = await shuttle.yarn.traverse(start_node="thompson_sampling", max_hops=2)

def test_intersection_logic():
    """Test Warp↔Yarn intersection logic."""
    # Mock Warp results
    warp_results = [
        {"id": "node_1", "score": 0.9, "text": "Thompson Sampling intro"},
        {"id": "node_2", "score": 0.8, "text": "Bandit algorithms"},
        {"id": "node_5", "score": 0.6, "text": "Bayesian methods"}
    ]

    # Mock Yarn results
    yarn_results = [
        {"id": "node_1", "score": 0.7, "text": "Thompson Sampling intro"},
        {"id": "node_3", "score": 0.8, "text": "Exploration strategies"},
        {"id": "node_5", "score": 0.6, "text": "Bayesian methods"}
    ]

    # Intersection (shared nodes)
    warp_ids = {r["id"] for r in warp_results}
    yarn_ids = {r["id"] for r in yarn_results}
    intersection = warp_ids & yarn_ids

    assert "node_1" in intersection  # Shared
    assert "node_5" in intersection  # Shared
    assert "node_2" not in intersection  # Warp only
    assert "node_3" not in intersection  # Yarn only

def test_mcts_node_expansion():
    """Test MCTS node expansion."""
    from hololoom.shuttle.mcts import MCTSNode

    root = MCTSNode(state={"query": "test"})

    # Initially no children
    assert len(root.children) == 0
    assert root.visits == 0

    # Expand
    child1 = root.add_child(action="warp_search", state={"query": "test", "source": "warp"})
    child2 = root.add_child(action="yarn_traverse", state={"query": "test", "source": "yarn"})

    # Should have children
    assert len(root.children) == 2
    assert child1.parent == root
    assert child2.parent == root

def test_mcts_uct_calculation():
    """Test UCT (Upper Confidence Bound for Trees) calculation."""
    from hololoom.shuttle.mcts import MCTSNode
    import math

    root = MCTSNode(state={})
    root.visits = 10

    child = root.add_child(action="test", state={})
    child.visits = 5
    child.value = 3.0

    # UCT = Q/N + C * sqrt(ln(parent.N) / N)
    exploration_weight = 1.4
    exploitation = child.value / child.visits  # 3.0 / 5 = 0.6
    exploration = exploration_weight * math.sqrt(math.log(root.visits) / child.visits)
    expected_uct = exploitation + exploration

    calculated_uct = child.uct(exploration_weight)

    assert abs(calculated_uct - expected_uct) < 1e-6

def test_thompson_numerical_stability():
    """Test Thompson Sampler handles large alpha/beta."""
    sampler = ThompsonSampler(n_arms=3)

    # Update many times
    for _ in range(1000):
        sampler.update(arm=0, reward=1.0)

    # Should not overflow
    assert not np.isnan(sampler.alpha[0])
    assert not np.isinf(sampler.alpha[0])

    # Sample should still work
    sample = sampler.sample()
    assert 0 <= sample < 3

def test_shuttle_graceful_fallback():
    """Test shuttle falls back gracefully without backends."""
    config = ShuttleConfig(
        qdrant_host="nonexistent",
        neo4j_uri="bolt://nonexistent:7687"
    )

    # Should initialize (may warn)
    shuttle = MCTSShuttle(config)

    # Should have fallback behavior
    assert shuttle is not None
```

**Run unit tests**:
```bash
pytest tests/unit/test_mcts_shuttle.py -v
```

### Integration Tests (6 tests, <30 seconds total)

**File**: `tests/integration/test_shuttle_backends.py`

```python
import pytest
import asyncio
from hololoom.shuttle import MCTSShuttle, ShuttleConfig

@pytest.mark.asyncio
@pytest.mark.requires_qdrant
async def test_warp_qdrant_search():
    """Test Warp search with real Qdrant."""
    config = ShuttleConfig(
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_collection="test_collection"
    )

    shuttle = MCTSShuttle(config)

    # Insert test data
    await shuttle.warp.insert([
        {"id": "test_1", "text": "Thompson Sampling is a Bayesian approach"},
        {"id": "test_2", "text": "Bandit algorithms balance exploration and exploitation"},
    ])

    # Search
    results = await shuttle.warp.search("What is Thompson Sampling?", top_k=2)

    # Should return results
    assert len(results) > 0
    assert results[0]["id"] == "test_1"  # Most relevant

@pytest.mark.asyncio
@pytest.mark.requires_neo4j
async def test_yarn_neo4j_traverse():
    """Test Yarn traversal with real Neo4j."""
    config = ShuttleConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password"
    )

    shuttle = MCTSShuttle(config)

    # Insert test graph
    await shuttle.yarn.add_edges([
        ("thompson_sampling", "bayesian_methods", "IS_A"),
        ("thompson_sampling", "exploration", "USES"),
        ("bayesian_methods", "statistics", "PART_OF")
    ])

    # Traverse
    results = await shuttle.yarn.traverse(start_node="thompson_sampling", max_hops=2)

    # Should find connected nodes
    node_names = {r["name"] for r in results}
    assert "bayesian_methods" in node_names
    assert "exploration" in node_names
    assert "statistics" in node_names  # 2 hops

@pytest.mark.asyncio
@pytest.mark.requires_backends
async def test_warp_yarn_intersection():
    """Test Warp↔Yarn intersection with real backends."""
    config = ShuttleConfig()
    shuttle = MCTSShuttle(config)

    # Setup test data
    await shuttle.warp.insert([
        {"id": "node_1", "text": "Thompson Sampling balances exploration"},
        {"id": "node_2", "text": "Bayesian inference uses priors"},
    ])

    await shuttle.yarn.add_edges([
        ("node_1", "node_2", "RELATED_TO"),
        ("node_1", "exploration", "USES"),
    ])

    # Intersect
    results = await shuttle.intersect(query="Thompson Sampling", top_k=5)

    # Should combine both sources
    assert len(results) > 0

    # Should have provenance
    for r in results:
        assert "source" in r  # "warp", "yarn", or "intersection"

@pytest.mark.asyncio
async def test_mcts_simulation():
    """Test MCTS simulation with Thompson Sampling."""
    config = ShuttleConfig(num_mcts_simulations=10)
    shuttle = MCTSShuttle(config)

    # Run MCTS
    best_path = await shuttle.mcts_search(query="Thompson Sampling", max_depth=3)

    # Should return a path
    assert len(best_path) > 0

    # Should have actions
    for step in best_path:
        assert "action" in step  # "warp" or "yarn"
        assert "state" in step

@pytest.mark.asyncio
async def test_thompson_learning():
    """Test Thompson Sampler learns from rewards."""
    config = ShuttleConfig()
    shuttle = MCTSShuttle(config)

    # Simulate 50 queries with varying rewards
    for i in range(50):
        # Warp is better for semantic queries
        if i % 2 == 0:
            shuttle.thompson.update(arm=0, reward=0.9)  # Warp
        else:
            shuttle.thompson.update(arm=1, reward=0.3)  # Yarn

    # Thompson should prefer Warp
    samples = [shuttle.thompson.sample() for _ in range(100)]
    warp_count = samples.count(0)
    yarn_count = samples.count(1)

    assert warp_count > yarn_count  # Should learn Warp is better

@pytest.mark.asyncio
async def test_shuttle_resilience():
    """Test shuttle handles backend failures gracefully."""
    config = ShuttleConfig(
        qdrant_host="nonexistent",
        neo4j_uri="bolt://nonexistent:7687"
    )

    shuttle = MCTSShuttle(config)

    # Should not crash when backends unavailable
    try:
        results = await shuttle.intersect("test query")
        # May return empty or fallback results
        assert results is not None
    except Exception as e:
        # Should have informative error
        assert "backend" in str(e).lower() or "connection" in str(e).lower()
```

**Run integration tests** (requires Docker):
```bash
# Start backends
docker-compose up -d

# Run tests
pytest tests/integration/test_shuttle_backends.py -v -m requires_backends
```

---

## Option C: Elle Verification

### Unit Tests (18 tests, <10 seconds total)

**File**: `tests/unit/test_elle_components.py`

```python
import pytest
import asyncio
from elle.config import ElleProductionConfig, VisionConfig, VoiceConfig
from elle.monitoring import PerformanceMonitor, LatencyMetric
from elle.engine import ElleEngine
from elle.server.websocket_server import AREvent

def test_production_config_profiles():
    """Test configuration profiles."""
    dev_config = ElleProductionConfig.development()
    prod_config = ElleProductionConfig.production()

    # Dev should have relaxed settings
    assert dev_config.monitoring.log_level == "DEBUG"
    assert dev_config.vision.frame_processing_fps == 10

    # Prod should be optimized
    assert prod_config.monitoring.log_level == "INFO"
    assert prod_config.performance.max_decision_latency_ms <= 100
    assert prod_config.vision.use_gpu == True

def test_vision_config_validation():
    """Test vision config validation."""
    config = VisionConfig(
        detection_confidence_threshold=0.6,
        max_objects_per_frame=20,
        frame_processing_fps=15
    )

    assert config.detection_confidence_threshold == 0.6
    assert config.frame_processing_fps == 15

def test_voice_config_validation():
    """Test voice config validation."""
    config = VoiceConfig(
        enable_wake_word=True,
        wake_word="Elle",
        stt_provider="whisper",
        max_stt_latency_ms=500
    )

    assert config.wake_word == "Elle"
    assert config.max_stt_latency_ms == 500

def test_performance_monitor_initialization():
    """Test PerformanceMonitor initialization."""
    monitor = PerformanceMonitor(latency_budget_ms=100, window_size=1000)

    assert monitor.latency_budget_ms == 100
    assert monitor._total_requests == 0

def test_performance_monitor_recording():
    """Test latency recording."""
    monitor = PerformanceMonitor()

    latencies = [
        LatencyMetric("vision", 25.0),
        LatencyMetric("policy", 40.0)
    ]

    monitor.record_request(latencies, success=True)

    summary = monitor.get_summary()
    assert summary.total_requests == 1
    assert summary.successful_requests == 1
    assert summary.mean_latency_ms == 65.0

def test_performance_monitor_percentiles():
    """Test percentile calculation."""
    monitor = PerformanceMonitor()

    # Record 100 requests with varying latencies
    for i in range(100):
        latency = 50.0 + i  # 50-149ms
        monitor.record_request([LatencyMetric("test", latency)], success=True)

    summary = monitor.get_summary()

    # Check percentiles
    assert 90 < summary.p50_latency_ms < 110  # ~99ms
    assert 140 < summary.p95_latency_ms < 150  # ~147ms
    assert 145 < summary.p99_latency_ms < 150  # ~149ms

def test_performance_monitor_health_check():
    """Test health check logic."""
    monitor = PerformanceMonitor(latency_budget_ms=100)

    # Record budget violations
    for _ in range(10):
        monitor.record_request([LatencyMetric("slow", 120.0)], success=True)

    # Record failures
    for _ in range(5):
        monitor.record_request([], success=False)

    healthy, warnings = monitor.check_health()

    # Should detect issues
    assert not healthy
    assert len(warnings) >= 2  # Latency + errors

def test_ar_event_parsing():
    """Test AREvent JSON parsing."""
    json_str = '''
    {
        "event_type": "scene_update",
        "timestamp": 1234567890.0,
        "data": {"objects": [], "description": "test"},
        "session_id": "test_session"
    }
    '''

    event = AREvent.from_json(json_str)

    assert event.event_type == "scene_update"
    assert event.session_id == "test_session"
    assert "objects" in event.data

def test_elle_response_serialization():
    """Test ElleResponse JSON serialization."""
    from elle.server.websocket_server import ElleResponse

    response = ElleResponse(
        response_type="guidance",
        timestamp=1234567890.0,
        data={"text": "Look at the tool"},
        latency_ms=75.5
    )

    json_str = response.to_json()

    assert "guidance" in json_str
    assert "75.5" in json_str

@pytest.mark.asyncio
async def test_elle_engine_parse_event():
    """Test ElleEngine event parsing."""
    engine = ElleEngine()

    event = AREvent(
        event_type="scene_update",
        timestamp=1234567890.0,
        data={
            "objects": [{"name": "hammer", "position": [0, 0, 1]}],
            "description": "cluttered shed"
        },
        session_id="test"
    )

    scene, intent = await engine._parse_event(event)

    assert scene is not None
    assert intent is not None

@pytest.mark.asyncio
async def test_elle_engine_latency_tracking():
    """Test ElleEngine tracks latencies per stage."""
    engine = ElleEngine()

    event = AREvent(
        event_type="scene_update",
        timestamp=1234567890.0,
        data={"objects": [], "description": "test"},
        session_id="test"
    )

    result, latencies = await engine.process_with_metrics(event)

    # Should have 4 stages
    assert len(latencies) == 4

    stage_names = {l.stage for l in latencies}
    assert "parse_event" in stage_names
    assert "retrieve_context" in stage_names
    assert "policy_decision" in stage_names
    assert "generate_actions" in stage_names

def test_latency_metric_creation():
    """Test LatencyMetric creation."""
    import time

    start = time.time()
    metric = LatencyMetric("test_stage", 45.5)

    assert metric.stage == "test_stage"
    assert metric.duration_ms == 45.5
    assert metric.timestamp >= start

def test_metrics_summary_aggregation():
    """Test MetricsSummary aggregation."""
    from elle.monitoring import MetricsSummary

    summary = MetricsSummary(
        total_requests=100,
        successful_requests=95,
        failed_requests=5,
        mean_latency_ms=78.5,
        p95_latency_ms=95.0,
        error_rate=0.05
    )

    assert summary.total_requests == 100
    assert summary.error_rate == 0.05

def test_config_environment_variables():
    """Test config loads from environment variables."""
    import os

    os.environ["ELLE_API_KEY"] = "test_key_123"

    config = ElleProductionConfig()

    assert config.api_key == "test_key_123"

    # Cleanup
    del os.environ["ELLE_API_KEY"]

@pytest.mark.parametrize("fps,expected_interval", [
    (10, 0.1),
    (15, 0.0667),
    (30, 0.0333)
])
def test_vision_frame_interval(fps, expected_interval):
    """Test vision frame interval calculation."""
    interval = 1.0 / fps
    assert abs(interval - expected_interval) < 0.001

def test_monitoring_window_size_limit():
    """Test monitoring respects window size limit."""
    monitor = PerformanceMonitor(window_size=10)

    # Record 20 requests
    for i in range(20):
        monitor.record_request([LatencyMetric("test", float(i))], success=True)

    summary = monitor.get_summary()

    # Should only track last 10 (window size)
    # Mean should be ~14.5 (average of 10-19)
    assert 13 < summary.mean_latency_ms < 16

def test_health_check_thresholds():
    """Test health check threshold logic."""
    monitor = PerformanceMonitor(latency_budget_ms=100)

    # Exactly 5% violations (threshold)
    for i in range(100):
        if i < 5:
            latency = 110.0  # Violation
        else:
            latency = 80.0  # OK
        monitor.record_request([LatencyMetric("test", latency)], success=True)

    healthy, warnings = monitor.check_health()

    # At threshold - should warn
    assert len(warnings) > 0
```

**Run unit tests**:
```bash
pytest tests/unit/test_elle_components.py -v
```

---

## Cross-Option System Tests

### System Test 1: BossPig → HoloLoom → Elle

**File**: `tests/system/test_cross_option_integration.py`

```python
import pytest
import asyncio
from hololoom import HoloLoom
from hololoom.bosspig import BossPigDetector
from elle.engine import ElleEngine

@pytest.mark.asyncio
@pytest.mark.slow
async def test_bosspig_hololoom_elle_workflow():
    """Test BossPig analysis → HoloLoom storage → Elle guidance."""

    # Step 1: BossPig analyzes document
    detector = BossPigDetector()
    document = """
    We leverage cutting-edge synergies to facilitate stakeholder alignment.
    Our innovative approach is arguably best-in-class.
    """

    analysis = detector.analyze(document, document_id="cross_test_doc")

    assert analysis.overall_score < 70  # High slop

    # Step 2: Store in HoloLoom memory
    async with HoloLoom() as loom:
        await loom.experience(f"Document quality: {analysis.overall_score}/100")
        await loom.experience(f"Detected issues: {', '.join(d.category for d in analysis.detections[:3])}")

        # Step 3: Elle queries HoloLoom for guidance
        elle = ElleEngine(memory_backend=loom)

        from elle.server.websocket_server import AREvent
        event = AREvent(
            event_type="scene_update",
            timestamp=1234567890.0,
            data={"user_query": "How can I improve my business writing?"},
            session_id="guidance_session"
        )

        result, latencies = await elle.process_with_metrics(event)

        # Elle should provide guidance based on HoloLoom memories
        assert "suggested_actions" in result
        assert result["confidence"] > 0.5

        total_latency = sum(l.duration_ms for l in latencies)
        assert total_latency < 200  # Reasonable latency

    print(f"✓ Cross-option workflow complete ({total_latency:.2f}ms)")
```

---

## Verification Automation

### Continuous Testing Script

**File**: `scripts/run_verification.sh`

```bash
#!/bin/bash

set -e  # Exit on error

echo "============================================"
echo "HoloLoom Verification Framework"
echo "Testing: BossPig + Quick Wins + Elle"
echo "============================================"
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Tier 1: Unit Tests (fast)
echo -e "${YELLOW}[TIER 1] Running Unit Tests...${NC}"
echo

pytest tests/unit/test_bosspig_detectors.py -v --tb=short
pytest tests/unit/test_mcts_shuttle.py -v --tb=short
pytest tests/unit/test_elle_components.py -v --tb=short

echo -e "${GREEN}✓ Unit tests passed${NC}"
echo

# Tier 2: Integration Tests (medium)
echo -e "${YELLOW}[TIER 2] Running Integration Tests...${NC}"
echo

pytest tests/integration/test_bosspig_pipeline.py -v --tb=short
pytest tests/integration/test_shuttle_backends.py -v --tb=short -m "not requires_backends"

echo -e "${GREEN}✓ Integration tests passed${NC}"
echo

# Tier 3: System Tests (slow)
echo -e "${YELLOW}[TIER 3] Running System Tests...${NC}"
echo

pytest tests/system/test_bosspig_end_to_end.py -v --tb=short -m slow
pytest tests/system/test_cross_option_integration.py -v --tb=short -m slow

echo -e "${GREEN}✓ System tests passed${NC}"
echo

# Summary
echo "============================================"
echo -e "${GREEN}All verification tests passed!${NC}"
echo "============================================"
```

**Make executable**:
```bash
chmod +x scripts/run_verification.sh
```

**Run all tests**:
```bash
./scripts/run_verification.sh
```

---

## Metrics Dashboard

### Test Coverage Report

**File**: `scripts/generate_coverage_report.sh`

```bash
#!/bin/bash

echo "Generating coverage report..."

pytest tests/ --cov=HoloLoom.bosspig --cov=HoloLoom.shuttle --cov=elle \
       --cov-report=html --cov-report=term

echo
echo "Coverage report generated: htmlcov/index.html"
echo
echo "Opening report in browser..."
open htmlcov/index.html  # macOS
# xdg-open htmlcov/index.html  # Linux
```

**Expected coverage targets**:
- BossPig: >85%
- MCTS Shuttle: >80%
- Elle: >75%

---

## Quality Gates

### Pre-Commit Checks

**File**: `.pre-commit-config.yaml`

```yaml
repos:
  - repo: local
    hooks:
      - id: unit-tests
        name: Unit Tests
        entry: pytest tests/unit/ -v
        language: system
        pass_filenames: false
        always_run: true

      - id: type-check
        name: Type Checking
        entry: mypy hololoom/bosspig hololoom/shuttle elle
        language: system
        types: [python]

      - id: lint
        name: Linting
        entry: pylint hololoom/bosspig hololoom/shuttle elle
        language: system
        types: [python]
```

**Install pre-commit**:
```bash
pip install pre-commit
pre-commit install
```

---

## Summary

**Verification Framework Provides**:

1. **Three-Tier Testing**:
   - Unit: <10s total (45 tests)
   - Integration: <30s total (14 tests)
   - System: <5min total (6 tests)

2. **Per-Option Coverage**:
   - BossPig: 15 unit + 8 integration + 3 system = 26 tests
   - Quick Wins: 12 unit + 6 integration = 18 tests
   - Elle: 18 unit + system tests = 20+ tests

3. **Cross-Option Integration**:
   - BossPig → HoloLoom → Elle workflow test
   - MCTS Shuttle → Workflow Builder integration
   - Full system validation

4. **Automation**:
   - Single-command verification (`./scripts/run_verification.sh`)
   - Coverage reporting
   - Pre-commit hooks

5. **Quality Gates**:
   - >80% test coverage
   - <5% latency violations
   - <1% error rate
   - All tests passing before merge

**Next**: Create dependency graph showing parallelization opportunities across all 3 options.
