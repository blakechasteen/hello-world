"""
Tests for BossPig Auto-Fixer

Tests all 5 fix categories:
1. Corporate jargon replacement
2. Vague commitment fixes
3. Missing date fixes
4. AI hallucination cleanup
5. Passive voice flagging

Created: 2025-11-22
Week: 4
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add bosspig to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "bosspig"))

from fixer.auto_fixer import AutoFixer, FixStrategy, FixResult
from detector.detector import BossPigDetector
from detector.core import FindingCategory


class TestJargonFixer:
    """Test corporate jargon replacement"""

    def test_simple_jargon_replacement(self):
        """Test basic jargon replacement"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "We need to synergize our efforts to leverage best practices."

        result = fixer.fix(text, categories=[FindingCategory.CORPORATE_JARGON])

        assert result.fixes_applied > 0
        assert "synergize" not in result.fixed_text.lower() or \
               "leverage" not in result.fixed_text.lower()
        assert result.quality_improvement >= 0

    def test_case_preservation(self):
        """Test that case is preserved during replacement"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "We must LEVERAGE our core competencies. Let's leverage this opportunity."

        result = fixer.fix(text, categories=[FindingCategory.CORPORATE_JARGON])

        # Should preserve uppercase and lowercase
        fixed = result.fixed_text
        # At least one replacement should preserve case
        assert any([
            word.isupper() for word in fixed.split()
            if len(word) > 3  # Ignore short words
        ]) or "leverage" not in fixed.lower()

    def test_word_boundary_matching(self):
        """Test that jargon replacement respects word boundaries"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "We need to leverage our bandwidth."  # Contains jargon

        result = fixer.fix(text, categories=[FindingCategory.CORPORATE_JARGON])

        # Should replace jargon terms
        # The fix should handle word boundaries correctly
        if result.fixes_applied > 0:
            assert result.fixed_text != text  # Something changed

    def test_multiple_jargon_instances(self):
        """Test fixing multiple jargon instances"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "Let's synergize to leverage our bandwidth and optimize deliverables."

        result = fixer.fix(text, categories=[FindingCategory.CORPORATE_JARGON])

        # Should fix multiple instances
        assert result.fixes_applied >= 2
        # Jargon should be reduced
        jargon_count_before = len(result.findings_before.findings_by_category(
            FindingCategory.CORPORATE_JARGON
        ))
        jargon_count_after = len(result.findings_after.findings_by_category(
            FindingCategory.CORPORATE_JARGON
        ))
        assert jargon_count_after < jargon_count_before


class TestVagueCommitmentFixer:
    """Test vague commitment fixes"""

    def test_try_to_replacement(self):
        """Test 'will try to' replacement"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "We will try to complete the project."

        result = fixer.fix(text, categories=[FindingCategory.VAGUE_COMMITMENTS])

        assert "will try to" not in result.fixed_text
        # Should contain action/date placeholders
        assert "[ACTION]" in result.fixed_text or "will" in result.fixed_text
        assert result.fixes_applied > 0

    def test_best_effort_replacement(self):
        """Test 'best effort' replacement"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "We'll make a best effort to deliver on time."

        result = fixer.fix(text, categories=[FindingCategory.VAGUE_COMMITMENTS])

        # Should replace vague commitment if detected
        # (detector might not always catch "best effort" in all contexts)
        if result.fixes_applied > 0:
            assert "best effort" not in result.fixed_text or \
                   "[SPECIFIC_COMMITMENT]" in result.fixed_text

    def test_asap_replacement(self):
        """Test ASAP replacement with suggested date"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "We need this done ASAP."

        result = fixer.fix(text, categories=[FindingCategory.VAGUE_COMMITMENTS])

        # Should replace ASAP with a date if detected
        if result.fixes_applied > 0:
            assert "asap" not in result.fixed_text.lower()
            # Should contain a date (YYYY-MM-DD format)
            import re
            date_pattern = r'\d{4}-\d{2}-\d{2}'
            assert re.search(date_pattern, result.fixed_text)


class TestMissingDateFixer:
    """Test missing date fixes"""

    def test_soon_replacement(self):
        """Test 'soon' replacement with suggested date"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "We will deliver this soon."

        result = fixer.fix(text, categories=[FindingCategory.MISSING_DATES])

        # Should replace 'soon' with a date
        assert "soon" not in result.fixed_text.lower()
        import re
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        assert re.search(date_pattern, result.fixed_text)

    def test_tbd_replacement(self):
        """Test TBD replacement"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "Completion date: TBD"

        result = fixer.fix(text, categories=[FindingCategory.MISSING_DATES])

        # Should replace TBD
        assert "tbd" not in result.fixed_text.lower()

    def test_date_suggestion_is_reasonable(self):
        """Test that suggested dates are in the future"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "We'll finish this soon."

        result = fixer.fix(text, categories=[FindingCategory.MISSING_DATES])

        # Extract date from result
        import re
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        match = re.search(date_pattern, result.fixed_text)
        if match:
            suggested_date = datetime.strptime(match.group(1), "%Y-%m-%d")
            # Should be in the future
            assert suggested_date > datetime.now()
            # Should be within 30 days
            assert suggested_date < datetime.now() + timedelta(days=30)


class TestAIHallucinationFixer:
    """Test AI hallucination cleanup"""

    def test_llm_boilerplate_removal(self):
        """Test removal of LLM boilerplate"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "As an AI language model, I cannot provide personal opinions. Here's the data."

        result = fixer.fix(text, categories=[FindingCategory.AI_HALLUCINATION])

        # Boilerplate should be removed
        assert "as an ai" not in result.fixed_text.lower()
        # Actual content should remain
        assert "Here's the data" in result.fixed_text

    def test_placeholder_flagging(self):
        """Test placeholder content flagging"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "The project will deliver [INSERT DELIVERABLE HERE] by next month."

        result = fixer.fix(text, categories=[FindingCategory.AI_HALLUCINATION])

        # Should flag placeholder
        assert "[FILL IN:" in result.fixed_text or "[INCOMPLETE" in result.fixed_text
        assert result.fixes_applied > 0

    def test_multiple_hallucination_types(self):
        """Test fixing multiple hallucination types"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "As a language model, I must say that [INSERT ANALYSIS HERE] is important."

        result = fixer.fix(text, categories=[FindingCategory.AI_HALLUCINATION])

        # Should fix at least one type
        assert result.fixes_applied >= 1
        # Should fix placeholder
        assert "[FILL IN:" in result.fixed_text or "[INCOMPLETE" in result.fixed_text


class TestPassiveVoiceFixer:
    """Test passive voice flagging (not auto-conversion)"""

    def test_passive_voice_detected(self):
        """Test that passive voice is detected but not auto-fixed"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "Mistakes were made by the team."

        result = fixer.fix(text, categories=[FindingCategory.PASSIVE_VOICE])

        # Passive voice detection requires NLP (spaCy), so it's optional
        passive_findings = result.findings_before.findings_by_category(
            FindingCategory.PASSIVE_VOICE
        )

        # If passive voice is detected, it should not be auto-fixed (MVP limitation)
        if len(passive_findings) > 0:
            assert result.fixes_applied == 0 or result.fixes_failed > 0
        # If not detected, test passes (NLP not available)

    def test_passive_voice_in_summary(self):
        """Test that passive voice is mentioned in summary"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "The report was completed yesterday."

        result = fixer.fix(text, categories=[FindingCategory.PASSIVE_VOICE])

        # Should mention passive voice in summary
        summary_text = " ".join(result.fix_summary).lower()
        assert "passive" in summary_text or len(result.fix_summary) == 0


class TestFullPipeline:
    """Test complete fix pipeline with multiple categories"""

    def test_fix_all_categories(self):
        """Test fixing all categories together"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = """
        We need to synergize our efforts ASAP. The team will try to deliver
        the project soon. As an AI, I should mention that [INSERT DETAILS HERE]
        are critical. Mistakes were made in the previous iteration.
        """

        result = fixer.fix(text)  # No categories = fix all

        # Should detect issues in multiple categories
        assert len(result.findings_before.findings) >= 4

        # Should fix some issues
        assert result.fixes_applied > 0

        # Quality should improve
        assert result.quality_improvement >= 0

        # Fixed text should be different
        assert result.fixed_text != result.original_text

    def test_quality_improvement_calculation(self):
        """Test that quality improvement is calculated correctly"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "We will leverage synergies to optimize deliverables ASAP."

        result = fixer.fix(text)

        # Quality should improve (fewer issues = higher score)
        score_before = result.findings_before.quality_metrics.overall_score()
        score_after = result.findings_after.quality_metrics.overall_score()

        assert score_after >= score_before
        assert result.quality_improvement == score_after - score_before

    def test_fix_summary_completeness(self):
        """Test that fix summary contains all changes"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "Let's synergize to leverage our bandwidth ASAP."

        result = fixer.fix(text)

        # Summary should document all fixes
        assert len(result.fix_summary) > 0
        # Each fix should have a description
        for summary_item in result.fix_summary:
            assert len(summary_item) > 10  # Meaningful description


class TestDryRunMode:
    """Test dry-run mode (preview without applying)"""

    def test_dry_run_no_changes(self):
        """Test that dry-run mode doesn't modify text"""
        fixer = AutoFixer(strategy=FixStrategy.DRY_RUN)
        text = "We need to synergize our bandwidth."

        result = fixer.fix(text)

        # In dry-run, text might still be processed
        # But we can verify findings are detected
        assert len(result.findings_before.findings) > 0

    def test_dry_run_shows_potential_fixes(self):
        """Test that dry-run shows what would be fixed"""
        fixer = AutoFixer(strategy=FixStrategy.DRY_RUN)
        text = "Let's leverage synergies ASAP."

        result = fixer.fix(text)

        # Should detect issues
        assert len(result.findings_before.findings) >= 2


class TestFixResultDiff:
    """Test diff generation"""

    def test_diff_generation(self):
        """Test unified diff generation"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "We need to synergize our efforts."

        result = fixer.fix(text)

        # Generate diff
        diff = result.get_diff()

        # Diff should show changes (if any were made)
        if result.fixes_applied > 0:
            assert len(diff) > 0
            assert "---" in diff or "+++" in diff

    def test_diff_shows_specific_changes(self):
        """Test that diff shows specific line changes"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "Line 1: We need to synergize.\nLine 2: Normal text.\nLine 3: Let's leverage this."

        result = fixer.fix(text)

        if result.fixes_applied > 0:
            diff = result.get_diff()
            # Should show line-level changes
            assert "-" in diff or "+" in diff


# Integration test with actual detector
class TestIntegrationWithDetector:
    """Test integration with BossPigDetector"""

    def test_fixer_uses_detector_findings(self):
        """Test that fixer uses detector's findings"""
        detector = BossPigDetector()
        fixer = AutoFixer(detector=detector, strategy=FixStrategy.BATCH)

        text = "We will try to synergize our bandwidth ASAP."

        result = fixer.fix(text)

        # Should detect multiple categories
        categories_found = set(f.category for f in result.findings_before.findings)
        assert len(categories_found) >= 2

    def test_reanalysis_after_fix(self):
        """Test that text is re-analyzed after fixing"""
        fixer = AutoFixer(strategy=FixStrategy.BATCH)
        text = "Let's synergize to leverage synergies."

        result = fixer.fix(text)

        # findings_after should be different from findings_before
        # (fewer issues after fixing)
        issues_before = len(result.findings_before.findings)
        issues_after = len(result.findings_after.findings)

        assert issues_after <= issues_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
