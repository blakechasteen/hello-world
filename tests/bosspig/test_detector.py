"""
BossPig Detector Unit Tests

Comprehensive test suite covering:
1. Corporate jargon detection
2. Vague commitments detection
3. Missing dates detection
4. AI hallucination markers detection
5. Passive voice detection
6. Scoring algorithm
7. Edge cases

Created: 2025-11-22
Status: Production Ready (Week 3)
Target: 20+ test cases
"""

import pytest
from bosspig.detector import (
    BossPigDetector,
    QualityScorer,
    FindingCategory,
    Severity,
    QualityMetrics,
    BossPigFindings
)


# ==================== DETECTOR 1: Corporate Jargon Tests ====================

def test_jargon_detection_basic():
    """Test basic jargon detection"""
    detector = BossPigDetector()
    text = "We need to synergize our core competencies to leverage the low-hanging fruit."

    results = detector.analyze(text)

    # Should detect: synergize, leverage, low-hanging fruit
    jargon_findings = results.findings_by_category(FindingCategory.CORPORATE_JARGON)
    assert len(jargon_findings) >= 2  # At minimum synergize and leverage


def test_jargon_case_insensitive():
    """Test jargon detection is case-insensitive"""
    detector = BossPigDetector()

    text1 = "We need to SYNERGIZE our efforts."
    text2 = "We need to synergize our efforts."
    text3 = "We need to Synergize our efforts."

    results1 = detector.analyze(text1)
    results2 = detector.analyze(text2)
    results3 = detector.analyze(text3)

    # All should detect jargon
    assert len(results1.findings) > 0
    assert len(results2.findings) > 0
    assert len(results3.findings) > 0


def test_jargon_word_boundary():
    """Test jargon detection respects word boundaries"""
    detector = BossPigDetector()

    # Should NOT match "synergize" in "hypersynergize"
    text = "The hypersynergized system works well."
    results = detector.analyze(text)

    # Should have fewer findings (word boundary respected)
    jargon_findings = [f for f in results.findings if "synergize" in f.text.lower()]
    assert len(jargon_findings) == 0  # Should not match partial word


def test_multiple_jargon_same_phrase():
    """Test detection of same jargon phrase multiple times"""
    detector = BossPigDetector()

    text = "We will leverage our resources to leverage new opportunities and leverage partnerships."
    results = detector.analyze(text)

    leverage_findings = [f for f in results.findings if "leverage" in f.text.lower()]
    assert len(leverage_findings) == 3  # Should find all 3 occurrences


# ==================== DETECTOR 2: Vague Commitments Tests ====================

def test_vague_commitments_detection():
    """Test detection of vague commitment language"""
    detector = BossPigDetector()

    text = """
    We will try to finish this project.
    Hopefully we can deliver next week.
    It would be nice if we could get approval.
    We should probably schedule a meeting.
    """

    results = detector.analyze(text)
    vague_findings = results.findings_by_category(FindingCategory.VAGUE_COMMITMENTS)

    # Should detect all 4 vague commitments
    assert len(vague_findings) >= 4


def test_vague_commitments_severity():
    """Test vague commitments are marked as CRITICAL"""
    detector = BossPigDetector()

    text = "We will try to complete this task."
    results = detector.analyze(text)

    vague_findings = results.findings_by_category(FindingCategory.VAGUE_COMMITMENTS)
    assert len(vague_findings) > 0
    assert all(f.severity == Severity.CRITICAL for f in vague_findings)


def test_specific_commitments_not_flagged():
    """Test that specific commitments are NOT flagged"""
    detector = BossPigDetector()

    text = "@alice will complete the migration by 2025-12-01."
    results = detector.analyze(text)

    vague_findings = results.findings_by_category(FindingCategory.VAGUE_COMMITMENTS)
    assert len(vague_findings) == 0  # No vague commitments


# ==================== DETECTOR 3: Missing Dates Tests ====================

def test_missing_dates_detection():
    """Test detection of vague date references"""
    detector = BossPigDetector()

    text = """
    We will complete this soon.
    The deadline is ASAP.
    Status is pending.
    Will determine TBD.
    Eventually we will finish.
    """

    results = detector.analyze(text)
    date_findings = results.findings_by_category(FindingCategory.MISSING_DATES)

    # Should detect: soon, ASAP, pending, TBD, eventually
    assert len(date_findings) >= 5


def test_specific_dates_not_flagged():
    """Test that specific dates are NOT flagged"""
    detector = BossPigDetector()

    text = """
    Deadline: 2025-12-01
    Meeting on November 25, 2025 at 3:00 PM
    Complete by Q1 2026
    """

    results = detector.analyze(text)
    date_findings = results.findings_by_category(FindingCategory.MISSING_DATES)

    assert len(date_findings) == 0  # No vague dates


def test_asap_variations():
    """Test detection of ASAP variations"""
    detector = BossPigDetector()

    text = """
    Please do this ASAP.
    Need this as soon as possible.
    Urgently needed asap.
    """

    results = detector.analyze(text)
    date_findings = results.findings_by_category(FindingCategory.MISSING_DATES)

    assert len(date_findings) >= 2  # ASAP and "as soon as possible"


# ==================== DETECTOR 4: AI Hallucination Tests ====================

def test_ai_hallucination_markers():
    """Test detection of AI boilerplate"""
    detector = BossPigDetector()

    text = """
    As an AI language model, I cannot provide specific recommendations.
    I don't have personal opinions, but here are some thoughts.
    """

    results = detector.analyze(text)
    ai_findings = results.findings_by_category(FindingCategory.AI_HALLUCINATION)

    # Should detect both AI markers
    assert len(ai_findings) >= 2


def test_placeholder_detection():
    """Test detection of placeholder text"""
    detector = BossPigDetector()

    text = """
    Product name: [INSERT PRODUCT NAME HERE]
    Features: [TO BE DETERMINED]
    Pricing: [TBD]
    See [TODO: add section reference]
    """

    results = detector.analyze(text)
    ai_findings = results.findings_by_category(FindingCategory.AI_HALLUCINATION)

    # Should detect all 4 placeholders
    assert len(ai_findings) >= 4


def test_ai_hallucination_severity():
    """Test AI hallucinations are marked as CRITICAL"""
    detector = BossPigDetector()

    text = "As an AI language model, I don't have opinions."
    results = detector.analyze(text)

    ai_findings = results.findings_by_category(FindingCategory.AI_HALLUCINATION)
    assert len(ai_findings) > 0
    assert all(f.severity == Severity.CRITICAL for f in ai_findings)


# ==================== DETECTOR 5: Passive Voice Tests ====================

def test_passive_voice_regex():
    """Test passive voice detection using regex (fallback)"""
    detector = BossPigDetector(enable_nlp=False)  # Force regex fallback

    text = """
    The report was completed by the team.
    Mistakes were made during the process.
    The decision has been determined.
    """

    results = detector.analyze(text)
    passive_findings = results.findings_by_category(FindingCategory.PASSIVE_VOICE)

    # Should detect at least 2 passive constructions
    assert len(passive_findings) >= 2


def test_active_voice_not_flagged():
    """Test that active voice is NOT flagged"""
    detector = BossPigDetector(enable_nlp=False)

    text = """
    Alice completed the report.
    The team made mistakes.
    Bob determined the solution.
    """

    results = detector.analyze(text)
    passive_findings = results.findings_by_category(FindingCategory.PASSIVE_VOICE)

    # Should have few or no passive voice findings
    assert len(passive_findings) == 0


# ==================== Scoring Algorithm Tests ====================

def test_quality_score_calculation():
    """Test quality score calculation"""
    detector = BossPigDetector()

    # Clean text (high score expected)
    clean_text = """
    @alice will complete the migration by 2025-12-01.
    The team achieved 95% uptime in Q3.
    Bob analyzed the data and found 3 issues.
    """

    # Dirty text (low score expected)
    dirty_text = """
    We will try to synergize our efforts soon.
    Hopefully we can leverage the paradigm shift.
    As an AI, I cannot provide opinions.
    [INSERT DETAILS HERE]
    """

    clean_results = detector.analyze(clean_text)
    dirty_results = detector.analyze(dirty_text)

    assert clean_results.quality_score > dirty_results.quality_score
    assert clean_results.grade in ["A", "B"]
    assert dirty_results.grade in ["D", "F"]


def test_grade_thresholds():
    """Test grade threshold boundaries"""
    # Test each grade threshold
    test_cases = [
        (QualityMetrics(1.0, 1.0, 1.0, 1.0, 1.0), "A"),  # Perfect score
        (QualityMetrics(0.85, 0.85, 0.85, 0.85, 0.85), "B"),  # Good
        (QualityMetrics(0.70, 0.70, 0.70, 0.70, 0.70), "C"),  # Fair
        (QualityMetrics(0.60, 0.60, 0.60, 0.60, 0.60), "D"),  # Poor
        (QualityMetrics(0.40, 0.40, 0.40, 0.40, 0.40), "F"),  # Critical
    ]

    for metrics, expected_grade in test_cases:
        assert metrics.grade() == expected_grade


def test_weighted_scoring():
    """Test that scoring weights are applied correctly"""
    # Clarity has 30% weight (most important)
    high_clarity = QualityMetrics(
        clarity_score=1.0,
        specificity_score=0.0,
        actionability_score=0.0,
        professionalism_score=0.0,
        completeness_score=0.0
    )

    # Completeness has 10% weight (least important)
    high_completeness = QualityMetrics(
        clarity_score=0.0,
        specificity_score=0.0,
        actionability_score=0.0,
        professionalism_score=0.0,
        completeness_score=1.0
    )

    # High clarity should score higher than high completeness
    assert high_clarity.overall_score() > high_completeness.overall_score()
    assert high_clarity.overall_score() == 30  # 30% weight
    assert high_completeness.overall_score() == 10  # 10% weight


# ==================== Edge Cases Tests ====================

def test_empty_text():
    """Test handling of empty text"""
    detector = BossPigDetector()

    results = detector.analyze("")

    # Should not crash, return valid (empty) results
    assert results.quality_score >= 0
    assert results.quality_score <= 100
    assert len(results.findings) == 0


def test_very_short_text():
    """Test handling of very short text"""
    detector = BossPigDetector()

    results = detector.analyze("OK")

    # Should not crash
    assert results.quality_score >= 0


def test_code_snippets_mixed_in():
    """Test that code snippets don't trigger false positives"""
    detector = BossPigDetector()

    text = """
    Here's the Python code:
    ```python
    def synergize_data(inputs):
        # Leverage the API
        return process(inputs)
    ```

    In our business plan, we should not synergize or leverage anything.
    """

    results = detector.analyze(text)

    # Should detect jargon in business text, but ideally not in code
    # (This is a known limitation - future enhancement)
    jargon_findings = results.findings_by_category(FindingCategory.CORPORATE_JARGON)
    assert len(jargon_findings) >= 1  # At least the business text jargon


def test_line_number_accuracy():
    """Test that line numbers are accurate"""
    detector = BossPigDetector()

    text = """Line 1
Line 2
Line 3 has synergize in it
Line 4"""

    results = detector.analyze(text)
    jargon_findings = results.findings_by_category(FindingCategory.CORPORATE_JARGON)

    assert len(jargon_findings) > 0
    # Line number should be 3
    assert jargon_findings[0].line_number == 3


def test_findings_have_suggestions():
    """Test that all findings have suggestions"""
    detector = BossPigDetector()

    text = "We will try to synergize soon."
    results = detector.analyze(text)

    # All findings should have non-empty suggestions
    for finding in results.findings:
        assert finding.suggestion
        assert len(finding.suggestion) > 0


def test_context_extraction():
    """Test that context is extracted for findings"""
    detector = BossPigDetector()

    text = "This is a long sentence with synergize in the middle of it."
    results = detector.analyze(text)

    jargon_findings = results.findings_by_category(FindingCategory.CORPORATE_JARGON)
    assert len(jargon_findings) > 0

    # Context should be non-empty
    assert jargon_findings[0].context
    assert "synergize" in jargon_findings[0].context


# ==================== Integration Tests ====================

def test_full_pipeline():
    """Test complete analysis pipeline"""
    detector = BossPigDetector()

    text = """
    # Q4 Strategic Initiative

    We need to synergize our core competencies to leverage the low-hanging fruit
    and move the needle on our key performance indicators.

    Action Items:
    - We will try to finish this soon
    - Hopefully we can circle back next week
    - Someone should touch base with stakeholders

    As an AI language model, I suggest: [INSERT PLAN HERE]
    """

    results = detector.analyze(text)

    # Should detect multiple categories
    assert len(results.findings_by_category(FindingCategory.CORPORATE_JARGON)) > 0
    assert len(results.findings_by_category(FindingCategory.VAGUE_COMMITMENTS)) > 0
    assert len(results.findings_by_category(FindingCategory.MISSING_DATES)) > 0
    assert len(results.findings_by_category(FindingCategory.AI_HALLUCINATION)) > 0

    # Score should be low (many issues)
    assert results.quality_score < 70  # Should be D or F

    # Should have critical findings
    assert results.critical_count() > 0


def test_scorer_recommendations():
    """Test that scorer generates recommendations"""
    detector = BossPigDetector()

    text = "We will try to synergize soon. [INSERT DETAILS HERE]"
    results = detector.analyze(text)

    breakdown = QualityScorer.calculate_detailed_score(results)
    recommendations = QualityScorer.get_improvement_recommendations(breakdown)

    # Should have recommendations
    assert len(recommendations) > 0

    # Recommendations should be sorted by priority
    priorities = [r["priority"] for r in recommendations]
    # Critical should come before high, high before medium
    assert priorities == sorted(priorities, key=lambda p: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(p, 99))


def test_score_report_generation():
    """Test score report generation"""
    detector = BossPigDetector()

    text = "We will try to synergize our efforts soon."
    results = detector.analyze(text)

    report = QualityScorer.generate_score_report(results)

    # Report should contain key information
    assert "Overall Score:" in report
    assert "Grade:" in report
    assert "Component Breakdown:" in report
    assert "Improvement Recommendations:" in report


# ==================== Performance Tests ====================

def test_processing_speed():
    """Test that processing is fast (<2 seconds for typical document)"""
    import time

    detector = BossPigDetector()

    # Typical document size (~500 words)
    text = " ".join(["We need to synergize our efforts."] * 100)

    start = time.time()
    results = detector.analyze(text)
    duration = time.time() - start

    # Should process in under 2 seconds
    assert duration < 2.0

    # Should still find issues
    assert len(results.findings) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
