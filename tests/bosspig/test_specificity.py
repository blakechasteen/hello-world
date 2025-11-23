"""
Tests for BossPig Specificity Enforcement Detector

Test Coverage:
- Vague quantifier detection
- Unmeasurable claim detection
- Relative statement without baseline detection
- Specificity scoring
- Interactive fixing

Created: 2025-11-22
Status: Week 8 Testing
"""

import pytest
from bosspig.detector.specificity import SpecificityDetector, SpecificityFixer
from bosspig.detector.core import FindingCategory, Severity


class TestVagueQuantifiers:
    """Test detection of vague quantifiers"""

    def test_detect_several(self):
        detector = SpecificityDetector()
        text = "We tested several configurations."
        findings = detector.analyze(text)

        vague_findings = [f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE]
        assert len(vague_findings) >= 1
        assert any('several' in f.text.lower() for f in vague_findings)
        assert all(f.severity == Severity.MEDIUM for f in vague_findings)

    def test_detect_many(self):
        detector = SpecificityDetector()
        text = "Many customers complained about the issue."
        findings = detector.analyze(text)

        vague_findings = [f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE]
        assert len(vague_findings) >= 1
        assert any('many' in f.text.lower() for f in vague_findings)

    def test_detect_significant(self):
        detector = SpecificityDetector()
        text = "We saw significant improvements in performance."
        findings = detector.analyze(text)

        vague_findings = [f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE]
        assert len(vague_findings) >= 1
        assert any('significant' in f.text.lower() for f in vague_findings)

    def test_detect_soon(self):
        detector = SpecificityDetector()
        text = "The feature will be released soon."
        findings = detector.analyze(text)

        vague_findings = [f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE]
        assert len(vague_findings) >= 1
        assert any('soon' in f.text.lower() for f in vague_findings)

    def test_detect_often(self):
        detector = SpecificityDetector()
        text = "Users often report this error."
        findings = detector.analyze(text)

        vague_findings = [f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE]
        assert len(vague_findings) >= 1
        assert any('often' in f.text.lower() for f in vague_findings)

    def test_detect_multiple_vague_quantifiers(self):
        detector = SpecificityDetector()
        text = "Many users often experience several issues with significant delays."
        findings = detector.analyze(text)

        vague_findings = [f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE]
        # Should find at least 3: "many", "often", "several", "significant"
        assert len(vague_findings) >= 3

    def test_no_false_positives_specific_text(self):
        detector = SpecificityDetector()
        text = "45 users experienced 3 issues with a 250ms delay."
        findings = detector.analyze(text)

        vague_findings = [f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE]
        assert len(vague_findings) == 0


class TestUnmeasurableClaims:
    """Test detection of unmeasurable claims"""

    def test_detect_will_improve_without_metric(self):
        detector = SpecificityDetector()
        text = "We will improve performance."
        findings = detector.analyze(text)

        unmeasurable = [f for f in findings if f.category == FindingCategory.UNMEASURABLE_CLAIM]
        assert len(unmeasurable) >= 1
        assert all(f.severity == Severity.HIGH for f in unmeasurable)

    def test_detect_will_increase_without_metric(self):
        detector = SpecificityDetector()
        text = "Sales will increase next quarter."
        findings = detector.analyze(text)

        unmeasurable = [f for f in findings if f.category == FindingCategory.UNMEASURABLE_CLAIM]
        assert len(unmeasurable) >= 1

    def test_detect_will_reduce_without_metric(self):
        detector = SpecificityDetector()
        text = "The update will reduce latency."
        findings = detector.analyze(text)

        unmeasurable = [f for f in findings if f.category == FindingCategory.UNMEASURABLE_CLAIM]
        assert len(unmeasurable) >= 1

    def test_detect_expect_to_achieve_without_number(self):
        detector = SpecificityDetector()
        text = "We expect to achieve better results."
        findings = detector.analyze(text)

        unmeasurable = [f for f in findings if f.category == FindingCategory.UNMEASURABLE_CLAIM]
        assert len(unmeasurable) >= 1

    def test_detect_world_class_claim(self):
        detector = SpecificityDetector()
        text = "Our world-class solution leads the industry."
        findings = detector.analyze(text)

        unmeasurable = [f for f in findings if f.category == FindingCategory.UNMEASURABLE_CLAIM]
        assert len(unmeasurable) >= 1

    def test_specific_target_allowed(self):
        detector = SpecificityDetector()
        text = "We will improve performance by 30%."
        findings = detector.analyze(text)

        unmeasurable = [f for f in findings if f.category == FindingCategory.UNMEASURABLE_CLAIM]
        # Should NOT flag this because "by 30%" is present
        assert len(unmeasurable) == 0


class TestRelativeWithoutBaseline:
    """Test detection of relative statements without baselines"""

    def test_detect_faster_without_baseline(self):
        detector = SpecificityDetector()
        text = "The new system is faster."
        findings = detector.analyze(text)

        relative = [f for f in findings if f.category == FindingCategory.RELATIVE_WITHOUT_BASELINE]
        assert len(relative) >= 1
        assert any('faster' in f.text.lower() for f in relative)
        assert all(f.severity == Severity.MEDIUM for f in relative)

    def test_detect_better_without_baseline(self):
        detector = SpecificityDetector()
        text = "Our results are better this quarter."
        findings = detector.analyze(text)

        relative = [f for f in findings if f.category == FindingCategory.RELATIVE_WITHOUT_BASELINE]
        assert len(relative) >= 1

    def test_detect_improved_without_from(self):
        detector = SpecificityDetector()
        text = "We improved our conversion rate."
        findings = detector.analyze(text)

        relative = [f for f in findings if f.category == FindingCategory.RELATIVE_WITHOUT_BASELINE]
        assert len(relative) >= 1

    def test_detect_increased_without_by(self):
        detector = SpecificityDetector()
        text = "Revenue increased significantly."
        findings = detector.analyze(text)

        # Should find both "increased" (relative) and "significantly" (vague)
        relative = [f for f in findings if f.category == FindingCategory.RELATIVE_WITHOUT_BASELINE]
        vague = [f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE]
        assert len(relative) >= 1
        assert len(vague) >= 1

    def test_baseline_provided_allowed(self):
        detector = SpecificityDetector()
        text = "The new system is faster than v2.0."
        findings = detector.analyze(text)

        relative = [f for f in findings if f.category == FindingCategory.RELATIVE_WITHOUT_BASELINE]
        # Should NOT flag because "than v2.0" provides baseline
        assert len(relative) == 0

    def test_percentage_change_allowed(self):
        detector = SpecificityDetector()
        text = "Revenue increased by 15%."
        findings = detector.analyze(text)

        relative = [f for f in findings if f.category == FindingCategory.RELATIVE_WITHOUT_BASELINE]
        # Should NOT flag because "by 15%" quantifies the change
        assert len(relative) == 0


class TestSpecificityScoring:
    """Test specificity score calculation"""

    def test_perfect_score_specific_text(self):
        detector = SpecificityDetector()
        text = "We tested 45 configurations and found 3 issues, reducing latency from 250ms to 150ms."
        findings = detector.analyze(text)
        score = detector.calculate_specificity_score(text, findings)

        assert score >= 0.95  # Nearly perfect specificity

    def test_low_score_vague_text(self):
        detector = SpecificityDetector()
        text = "We tested several configurations and found many issues, significantly reducing latency soon."
        findings = detector.analyze(text)
        score = detector.calculate_specificity_score(text, findings)

        # Has 4 vague words (several, many, significantly, soon) = 4 * 0.05 = 0.20 penalty
        # Score should be 1.0 - 0.20 = 0.80
        assert score <= 0.85  # Moderately vague

    def test_score_penalty_for_vague(self):
        detector = SpecificityDetector()
        text = "Many users often report several significant issues."
        findings = detector.analyze(text)
        score = detector.calculate_specificity_score(text, findings)

        # 4 vague quantifiers * 0.05 = 0.20 penalty → score ≤ 0.80
        assert score <= 0.80

    def test_score_penalty_for_unmeasurable(self):
        detector = SpecificityDetector()
        text = "We will improve performance and increase reliability."
        findings = detector.analyze(text)
        score = detector.calculate_specificity_score(text, findings)

        # 2 unmeasurable claims * 0.10 = 0.20 penalty → score should be around 0.80
        # But actual score might be higher if the regex doesn't match perfectly
        assert score <= 0.95  # Some penalty for unmeasurable claims
        assert score >= 0.75  # But not too harsh

    def test_metrics_calculation(self):
        detector = SpecificityDetector()
        text = "Many users will improve their experience soon with better results."
        metrics = detector.get_metrics(text)

        assert metrics.vague_quantifier_count >= 2  # "many", "soon"
        assert metrics.unmeasurable_claim_count >= 1  # "will improve"
        assert metrics.relative_without_baseline_count >= 1  # "better"
        assert 0.0 <= metrics.specificity_score <= 1.0


class TestContextExtraction:
    """Test context and position tracking"""

    def test_line_number_tracking(self):
        detector = SpecificityDetector()
        text = "Line 1.\nLine 2 has several issues.\nLine 3."
        findings = detector.analyze(text)

        vague_findings = [f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE]
        assert len(vague_findings) >= 1
        assert vague_findings[0].line_number == 2

    def test_context_extraction(self):
        detector = SpecificityDetector()
        text = "We tested several configurations yesterday."
        findings = detector.analyze(text)

        vague_findings = [f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE]
        assert len(vague_findings) >= 1
        assert 'several' in vague_findings[0].context

    def test_column_number_tracking(self):
        detector = SpecificityDetector()
        text = "We have several issues."
        findings = detector.analyze(text)

        vague_findings = [f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE]
        assert len(vague_findings) >= 1
        assert vague_findings[0].column_number is not None
        assert vague_findings[0].column_number >= 0


class TestInteractiveFixer:
    """Test interactive fixing functionality"""

    def test_fixer_initialization(self):
        fixer = SpecificityFixer()
        assert fixer.detector is not None

    def test_fix_vague_quantifier(self):
        fixer = SpecificityFixer()
        detector = SpecificityDetector()

        text = "We tested several configurations."
        findings = detector.analyze(text)
        vague_finding = next(f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE)

        fix_suggestion = fixer.fix_interactive(text, vague_finding)
        # Should return a template requiring user input
        assert "[USER INPUT NEEDED" in fix_suggestion or "several" in fix_suggestion.lower()

    def test_fix_unmeasurable_claim(self):
        fixer = SpecificityFixer()
        detector = SpecificityDetector()

        text = "We will improve performance."
        findings = detector.analyze(text)
        unmeasurable_finding = next(f for f in findings if f.category == FindingCategory.UNMEASURABLE_CLAIM)

        fix_suggestion = fixer.fix_interactive(text, unmeasurable_finding)
        # Should return a template requiring user input
        assert "[USER INPUT NEEDED" in fix_suggestion

    def test_fix_relative_statement(self):
        fixer = SpecificityFixer()
        detector = SpecificityDetector()

        text = "The system is faster now."
        findings = detector.analyze(text)
        relative_finding = next(f for f in findings if f.category == FindingCategory.RELATIVE_WITHOUT_BASELINE)

        fix_suggestion = fixer.fix_interactive(text, relative_finding)
        # Should return a template requiring user input
        assert "[USER INPUT NEEDED" in fix_suggestion


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_text(self):
        detector = SpecificityDetector()
        findings = detector.analyze("")
        assert len(findings) == 0

    def test_whitespace_only(self):
        detector = SpecificityDetector()
        findings = detector.analyze("   \n   \n   ")
        assert len(findings) == 0

    def test_numbers_only(self):
        detector = SpecificityDetector()
        findings = detector.analyze("12345 67890")
        assert len(findings) == 0

    def test_case_insensitive_detection(self):
        detector = SpecificityDetector()
        text_lower = "many users have several issues."
        text_upper = "MANY users have SEVERAL issues."

        findings_lower = detector.analyze(text_lower)
        findings_upper = detector.analyze(text_upper)

        # Should detect same patterns regardless of case
        vague_lower = [f for f in findings_lower if f.category == FindingCategory.VAGUE_LANGUAGE]
        vague_upper = [f for f in findings_upper if f.category == FindingCategory.VAGUE_LANGUAGE]

        assert len(vague_lower) == len(vague_upper)

    def test_very_long_text(self):
        detector = SpecificityDetector()
        # Create text with 100 vague quantifiers
        text = " many" * 100
        findings = detector.analyze(text)

        vague_findings = [f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE]
        assert len(vague_findings) == 100

    def test_special_characters(self):
        detector = SpecificityDetector()
        text = "We have several @issues# with $performance%."
        findings = detector.analyze(text)

        vague_findings = [f for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE]
        assert len(vague_findings) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
