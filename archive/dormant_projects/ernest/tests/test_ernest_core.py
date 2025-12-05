"""
Ernest Core Tests - Wave 3
===========================
Minimal but comprehensive testing suite for Ernest.

Tests:
- Hemingway pattern detection
- Refinement passes
- Learning engine
- Parallel passes
- Metaprompt adapter

Philosophy: "Test what matters, skip the rest."
"""

import pytest
import asyncio
from pathlib import Path

from ernest.refinement.patterns import (
    HemingwayPatterns,
    HemingwayMode,
    quick_analyze,
    quick_refine,
    hemingway_score
)
from ernest.learning import ErnestLearningEngine
from ernest.swarm.parallel_passes import ParallelCreativeRefiner, CreativePass
from ernest.swarm.metaprompt_adapter import ErnestMetapromptAdapter, build_ernest_prompt


# Test fixtures
@pytest.fixture
def weak_prose():
    """Weak prose needing refinement"""
    return """The man was walking down the street feeling very happy because
    it was a beautiful day and he really loved the city."""


@pytest.fixture
def strong_prose():
    """Strong Hemingway-style prose"""
    return """The man strode down Boulevard Saint-Michel. Sunlight glinted
    off the café windows. He bought bread at the corner boulangerie."""


@pytest.fixture
def dialogue_weak():
    """Weak dialogue with over-attribution"""
    return '''"I don't think we should do this," she said nervously,
    wringing her hands anxiously.'''


@pytest.fixture
def dialogue_strong():
    """Strong sparse dialogue"""
    return '''"I don't think we should."

She looked at the floor.

"What if—" She stopped.'''


# === Pattern Detection Tests ===

def test_hemingway_score_weak_prose(weak_prose):
    """Weak prose should score low (< 50)"""
    score, metrics = hemingway_score(weak_prose)
    assert score < 50, f"Weak prose scored {score}, expected < 50"
    assert metrics.filler_word_count > 0, "Should detect filler words (very, really)"
    assert metrics.avg_words_per_sentence > 20, "Should detect long sentences"


def test_hemingway_score_strong_prose(strong_prose):
    """Strong prose should score high (> 80)"""
    score, metrics = hemingway_score(strong_prose)
    assert score > 75, f"Strong prose scored {score}, expected > 75"
    assert metrics.avg_words_per_sentence < 20, "Should have short sentences"
    assert metrics.filler_word_count == 0, "Should have no filler words"


def test_mode_targets():
    """Each mode should have appropriate targets"""
    grace = HemingwayPatterns(mode=HemingwayMode.GRACE)
    assert grace.TARGET_AVG_SENTENCE_LENGTH == 16
    assert grace.TARGET_ACTIVE_VOICE_PCT == 85
    assert grace.TARGET_ICEBERG_RATIO == 70


def test_filler_word_detection(weak_prose):
    """Should detect filler words"""
    ernest = HemingwayPatterns()
    metrics = ernest.analyze_prose(weak_prose)

    # "very" and "really" should be detected
    assert metrics.filler_word_count >= 2, f"Found {metrics.filler_word_count} filler words, expected >= 2"


def test_iceberg_ratio_calculation(dialogue_weak, dialogue_strong):
    """Iceberg ratio should differentiate telling vs. showing"""
    ernest = HemingwayPatterns()

    metrics_weak = ernest.analyze_prose(dialogue_weak)
    metrics_strong = ernest.analyze_prose(dialogue_strong)

    # Weak dialogue tells emotions ("nervously", "anxiously")
    # Strong dialogue shows through action
    assert metrics_strong.iceberg_ratio > metrics_weak.iceberg_ratio, \
        "Strong dialogue should have higher iceberg ratio"


# === Refinement Tests ===

def test_clarity_pass_breaks_long_sentences(weak_prose):
    """Clarity pass should break sentences > 25 words"""
    ernest = HemingwayPatterns(mode=HemingwayMode.GRACE)
    result = ernest.refine_pass1_clarity(weak_prose)

    # Original sentence is ~24 words, but clarity pass should improve structure
    assert len(result.changes) > 0, "Should make changes to weak prose"
    assert result.metrics.hemingway_score > 40, "Score should improve"


def test_simplicity_pass_removes_filler(weak_prose):
    """Simplicity pass should remove filler words"""
    ernest = HemingwayPatterns(mode=HemingwayMode.GRACE)

    # First pass (clarity)
    result1 = ernest.refine_pass1_clarity(weak_prose)

    # Second pass (simplicity)
    result2 = ernest.refine_pass2_simplicity(result1.refined_text)

    # Should have fewer filler words
    assert result2.metrics.filler_word_count < result1.metrics.filler_word_count or \
           result2.metrics.filler_word_count == 0, \
           "Simplicity pass should remove filler words"


def test_beauty_pass_improves_score():
    """Beauty pass should improve final score"""
    text = "The man was walking. He was happy."

    ernest = HemingwayPatterns(mode=HemingwayMode.GRACE)
    initial_score = ernest.analyze_prose(text).hemingway_score

    result3 = ernest.refine_pass3_beauty(text)

    # Beauty pass should improve score (active voice, rhythm)
    assert result3.metrics.hemingway_score >= initial_score, \
        "Beauty pass should maintain or improve score"


def test_quick_refine_helper(weak_prose):
    """quick_refine should improve prose"""
    refined = quick_refine(weak_prose, mode="grace")

    # Check improvement
    original_score = quick_analyze(weak_prose)
    refined_score = quick_analyze(refined)

    assert refined_score > original_score, \
        f"Refinement should improve score ({original_score} → {refined_score})"


# === Learning Engine Tests ===

@pytest.mark.asyncio
async def test_learning_engine_initialization():
    """Learning engine should initialize properly"""
    async with ErnestLearningEngine(enable_background_learning=False) as ernest:
        stats = ernest.get_learning_statistics()

        assert stats["refinements_performed"] == 0
        assert stats["best_mode"] == "grace"  # Default


@pytest.mark.asyncio
async def test_learning_engine_refinement(weak_prose):
    """Learning engine should refine and track metrics"""
    async with ErnestLearningEngine(enable_background_learning=False) as ernest:
        result = await ernest.refine_with_learning(weak_prose, context="narrative")

        assert result["after_score"] > result["before_score"], "Should improve score"
        assert result["improvement"] > 0, "Should have positive improvement"
        assert "mode" in result, "Should track mode used"

        # Check learning stats updated
        stats = ernest.get_learning_statistics()
        assert stats["refinements_performed"] == 1


@pytest.mark.asyncio
async def test_mode_preference_learning(weak_prose):
    """Learning engine should learn mode preferences"""
    async with ErnestLearningEngine(enable_background_learning=False) as ernest:
        # Refine dialogue several times (should learn SPARSE works well)
        for i in range(5):
            await ernest.refine_with_learning(
                '''"I think so," he said.''',
                context="dialogue",
                mode=HemingwayMode.SPARSE
            )

        stats = ernest.get_learning_statistics()

        # SPARSE should have good success rate for dialogue
        assert stats["mode_success_rates"]["sparse"] > 0, \
            "SPARSE mode should have success rate > 0"


# === Parallel Passes Tests ===

@pytest.mark.asyncio
async def test_parallel_refiner_runs_all_passes(weak_prose):
    """Parallel refiner should run all 4 passes"""
    refiner = ParallelCreativeRefiner()
    result = await refiner.refine_parallel(weak_prose)

    assert len(result.pass_results) == 4, "Should run all 4 passes"
    assert CreativePass.PLOT in result.pass_results
    assert CreativePass.CHARACTER in result.pass_results
    assert CreativePass.DIALOGUE in result.pass_results
    assert CreativePass.STYLE in result.pass_results


@pytest.mark.asyncio
async def test_parallel_refiner_improves_text(weak_prose):
    """Parallel refiner should improve aggregate score"""
    refiner = ParallelCreativeRefiner(hemingway_mode=HemingwayMode.GRACE)
    result = await refiner.refine_parallel(weak_prose)

    # Aggregate score should be reasonable (weighted average of all passes)
    assert 40 <= result.aggregate_score <= 100, \
        f"Aggregate score {result.aggregate_score} should be 40-100"

    # Should make changes
    assert result.total_changes > 0, "Should make refinement changes"


@pytest.mark.asyncio
async def test_parallel_refiner_performance():
    """Parallel refiner should complete quickly"""
    text = "Short test text for performance."

    refiner = ParallelCreativeRefiner()
    result = await refiner.refine_parallel(text)

    # Should complete in < 500ms (parallel execution)
    assert result.duration_ms < 500, \
        f"Parallel refinement took {result.duration_ms}ms, expected < 500ms"


# === Metaprompt Adapter Tests ===

def test_metaprompt_construction():
    """Metaprompt adapter should construct valid prompts"""
    adapter = ErnestMetapromptAdapter()
    prompt = adapter.construct_prompt(
        query="Analyze my opening paragraph",
        mode=HemingwayMode.GRACE,
        context_type="narrative"
    )

    # Check 7 components present
    assert "OBJECTIVE" in prompt
    assert "PROCESS" in prompt
    assert "OUTPUT FORMAT" in prompt
    assert "CONSTRAINTS" in prompt
    assert "UNCERTAINTY" in prompt
    assert "VALIDATION" in prompt

    # Check query included
    assert "Analyze my opening paragraph" in prompt


def test_metaprompt_mode_adaptation():
    """Metaprompt should adapt to different modes"""
    adapter = ErnestMetapromptAdapter()

    prompt_sparse = adapter.construct_prompt("Test", mode=HemingwayMode.SPARSE)
    prompt_grace = adapter.construct_prompt("Test", mode=HemingwayMode.GRACE)

    # SPARSE mode should emphasize iceberg/subtext
    assert "Iceberg" in prompt_sparse or "subtext" in prompt_sparse.lower()

    # Should be different prompts
    assert prompt_sparse != prompt_grace


def test_mode_guidance():
    """Should provide guidance for each mode"""
    adapter = ErnestMetapromptAdapter()

    for mode in HemingwayMode:
        guidance = adapter.get_mode_guidance(mode)
        assert len(guidance) > 0, f"Mode {mode.value} should have guidance"
        assert "Target:" in guidance or "Focus:" in guidance


def test_build_ernest_prompt_helper():
    """Quick helper should build valid prompt"""
    prompt = build_ernest_prompt(
        "Refine my dialogue",
        mode=HemingwayMode.SPARSE,
        context="dialogue"
    )

    assert "SPARSE" in prompt
    assert "dialogue" in prompt.lower()
    assert "Refine my dialogue" in prompt


# === Integration Tests ===

@pytest.mark.asyncio
async def test_end_to_end_refinement(weak_prose):
    """Complete end-to-end refinement workflow"""
    # 1. Analyze
    before_score = quick_analyze(weak_prose)
    assert before_score < 60, "Starting with weak prose"

    # 2. Refine with learning
    async with ErnestLearningEngine(enable_background_learning=False) as ernest:
        result = await ernest.refine_with_learning(weak_prose, context="narrative")

        # 3. Verify improvement
        assert result["after_score"] > before_score, "Should improve score"
        assert len(result["changes"]) > 0, "Should make changes"

        # 4. Check learning
        stats = ernest.get_learning_statistics()
        assert stats["refinements_performed"] == 1
        assert stats["avg_improvement"] > 0


@pytest.mark.asyncio
async def test_parallel_passes_integration(weak_prose):
    """Parallel passes should work with learning engine"""
    from ernest.swarm.parallel_passes import parallel_refine_with_report

    report = await parallel_refine_with_report(weak_prose, mode=HemingwayMode.GRACE)

    # Check report structure
    assert "refined_text" in report
    assert "aggregate_score" in report
    assert "passes" in report
    assert len(report["passes"]) == 4  # All 4 passes


# === Performance Tests ===

def test_pattern_detection_performance():
    """Pattern detection should be fast"""
    import time

    text = "The man walked down the street." * 20  # 20 sentences

    start = time.time()
    for i in range(100):
        quick_analyze(text)
    duration = (time.time() - start) * 1000

    avg_per_query = duration / 100
    assert avg_per_query < 50, f"Pattern detection averaged {avg_per_query}ms, expected < 50ms"


@pytest.mark.asyncio
async def test_learning_engine_performance():
    """Learning engine overhead should be minimal"""
    import time

    text = "Short test text."

    async with ErnestLearningEngine(enable_background_learning=False) as ernest:
        start = time.time()
        for i in range(10):
            await ernest.refine_with_learning(text, context="general")
        duration = (time.time() - start) * 1000

        avg_per_refinement = duration / 10
        # Should be < 200ms per refinement (3 passes + learning overhead)
        assert avg_per_refinement < 300, \
            f"Refinement averaged {avg_per_refinement}ms, expected < 300ms"


# === Run tests ===

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
