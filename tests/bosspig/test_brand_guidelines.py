"""
Tests for BossPig Brand Guidelines Compliance Detector

Test Coverage:
- Brand capitalization enforcement
- Prohibited terms detection
- Preferred terminology checking
- Tone violation detection
- Compliance scoring

Created: 2025-11-22
Status: Week 9 Testing
"""

import pytest
from pathlib import Path
from bosspig.detector.brand_guidelines import (
    BrandGuidelinesDetector,
    calculate_brand_compliance_score,
    BrandConfig
)
from bosspig.detector.core import FindingCategory, Severity


class TestBrandCapitalization:
    """Test detection of incorrect brand capitalization"""

    def test_detect_techcorp_lowercase(self):
        detector = BrandGuidelinesDetector()
        text = "We use techcorp's services."
        findings = detector.analyze(text)

        cap_findings = [f for f in findings if f.category == FindingCategory.BRAND_CAPITALIZATION]
        assert len(cap_findings) >= 1
        assert any('techcorp' in f.text.lower() for f in cap_findings)
        assert any('TechCorp' in f.suggestion for f in cap_findings)

    def test_detect_cloudsync_wrong_case(self):
        detector = BrandGuidelinesDetector()
        text = "Use cloudsync for file storage."
        findings = detector.analyze(text)

        cap_findings = [f for f in findings if f.category == FindingCategory.BRAND_CAPITALIZATION]
        assert len(cap_findings) >= 1
        assert any('CloudSync' in f.suggestion for f in cap_findings)

    def test_detect_github_lowercase(self):
        detector = BrandGuidelinesDetector()
        text = "Push your code to github."
        findings = detector.analyze(text)

        cap_findings = [f for f in findings if f.category == FindingCategory.BRAND_CAPITALIZATION]
        assert len(cap_findings) >= 1
        assert any('GitHub' in f.suggestion for f in cap_findings)

    def test_correct_capitalization_allowed(self):
        detector = BrandGuidelinesDetector()
        text = "TechCorp's CloudSync and GitHub integration work seamlessly."
        findings = detector.analyze(text)

        cap_findings = [f for f in findings if f.category == FindingCategory.BRAND_CAPITALIZATION]
        assert len(cap_findings) == 0

    def test_multiple_capitalization_errors(self):
        detector = BrandGuidelinesDetector()
        text = "techcorp, cloudsync, and github are great."
        findings = detector.analyze(text)

        cap_findings = [f for f in findings if f.category == FindingCategory.BRAND_CAPITALIZATION]
        assert len(cap_findings) >= 3

    def test_case_insensitive_matching(self):
        detector = BrandGuidelinesDetector()
        text = "TECHCORP and TechCorp and techcorp"
        findings = detector.analyze(text)

        cap_findings = [f for f in findings if f.category == FindingCategory.BRAND_CAPITALIZATION]
        # Should detect TECHCORP and techcorp (not TechCorp which is correct)
        assert len(cap_findings) >= 2


class TestProhibitedTerms:
    """Test detection of prohibited terms"""

    def test_detect_cheap(self):
        detector = BrandGuidelinesDetector()
        text = "Our service is cheap and effective."
        findings = detector.analyze(text)

        prohibited = [f for f in findings if f.category == FindingCategory.PROHIBITED_TERM]
        assert len(prohibited) >= 1
        assert any('cheap' in f.text.lower() for f in prohibited)
        assert any('cost-effective' in f.suggestion for f in prohibited)

    def test_detect_buy_now(self):
        detector = BrandGuidelinesDetector()
        text = "Buy now to get started!"
        findings = detector.analyze(text)

        prohibited = [f for f in findings if f.category == FindingCategory.PROHIBITED_TERM]
        assert len(prohibited) >= 1
        assert any('buy now' in f.text.lower() for f in prohibited)

    def test_detect_guarantee(self):
        detector = BrandGuidelinesDetector()
        text = "We guarantee results."
        findings = detector.analyze(text)

        prohibited = [f for f in findings if f.category == FindingCategory.PROHIBITED_TERM]
        assert len(prohibited) >= 1
        assert any('guarantee' in f.text.lower() for f in prohibited)
        # Should be CRITICAL severity
        assert any(f.severity == Severity.CRITICAL for f in prohibited if 'guarantee' in f.text.lower())

    def test_detect_best_claim(self):
        detector = BrandGuidelinesDetector()
        text = "We are the best solution on the market."
        findings = detector.analyze(text)

        prohibited = [f for f in findings if f.category == FindingCategory.PROHIBITED_TERM]
        assert len(prohibited) >= 1
        assert any('best' in f.text.lower() for f in prohibited)

    def test_allow_best_practices(self):
        detector = BrandGuidelinesDetector()
        text = "Follow best practices for deployment."
        findings = detector.analyze(text)

        prohibited = [f for f in findings if f.category == FindingCategory.PROHIBITED_TERM]
        # "best practices" should be allowed (negative lookahead in pattern)
        assert not any('best' in f.text.lower() for f in prohibited)

    def test_detect_unlimited(self):
        detector = BrandGuidelinesDetector()
        text = "Get unlimited storage today."
        findings = detector.analyze(text)

        prohibited = [f for f in findings if f.category == FindingCategory.PROHIBITED_TERM]
        assert len(prohibited) >= 1
        assert any('unlimited' in f.text.lower() for f in prohibited)

    def test_detect_free_but_allow_exceptions(self):
        detector = BrandGuidelinesDetector()

        # Should detect standalone "free"
        text1 = "Our service is free."
        findings1 = detector.analyze(text1)
        prohibited1 = [f for f in findings1 if f.category == FindingCategory.PROHIBITED_TERM and 'free' in f.text.lower()]
        assert len(prohibited1) >= 1

        # Should allow "free trial"
        text2 = "Start your free trial today."
        findings2 = detector.analyze(text2)
        prohibited2 = [f for f in findings2 if f.category == FindingCategory.PROHIBITED_TERM and 'free trial' in f.text.lower()]
        assert len(prohibited2) == 0  # "free trial" is allowed

    def test_multiple_prohibited_terms(self):
        detector = BrandGuidelinesDetector()
        text = "Buy now to get cheap, unlimited storage guaranteed!"
        findings = detector.analyze(text)

        prohibited = [f for f in findings if f.category == FindingCategory.PROHIBITED_TERM]
        # Should find: buy now, cheap, unlimited, guaranteed
        assert len(prohibited) >= 4


class TestPreferredTerminology:
    """Test detection of non-preferred terms"""

    def test_detect_customer_prefer_client(self):
        detector = BrandGuidelinesDetector()
        text = "Our customers love the service."
        findings = detector.analyze(text)

        non_preferred = [f for f in findings if f.category == FindingCategory.NON_PREFERRED_TERM]
        assert len(non_preferred) >= 1
        assert any('customer' in f.text.lower() for f in non_preferred)
        assert any('clients' in f.suggestion for f in non_preferred)

    def test_detect_bugs_prefer_issues(self):
        detector = BrandGuidelinesDetector()
        text = "We fixed several bugs in the last release."
        findings = detector.analyze(text)

        non_preferred = [f for f in findings if f.category == FindingCategory.NON_PREFERRED_TERM]
        assert len(non_preferred) >= 1
        assert any('bug' in f.text.lower() for f in non_preferred)
        assert any('issues' in f.suggestion for f in non_preferred)

    def test_detect_login_prefer_sign_in(self):
        detector = BrandGuidelinesDetector()
        text = "Please login to continue."
        findings = detector.analyze(text)

        non_preferred = [f for f in findings if f.category == FindingCategory.NON_PREFERRED_TERM]
        assert len(non_preferred) >= 1
        assert any('login' in f.text.lower() for f in non_preferred)
        assert any('sign in' in f.suggestion for f in non_preferred)

    def test_detect_purchase_prefer_subscription(self):
        detector = BrandGuidelinesDetector()
        text = "Complete your purchase today."
        findings = detector.analyze(text)

        non_preferred = [f for f in findings if f.category == FindingCategory.NON_PREFERRED_TERM]
        assert len(non_preferred) >= 1
        assert any('purchase' in f.text.lower() for f in non_preferred)
        assert any('subscription' in f.suggestion for f in non_preferred)

    def test_detect_click_here(self):
        detector = BrandGuidelinesDetector()
        text = "Click here to learn more."
        findings = detector.analyze(text)

        non_preferred = [f for f in findings if f.category == FindingCategory.NON_PREFERRED_TERM]
        assert len(non_preferred) >= 1
        assert any('click here' in f.text.lower() for f in non_preferred)


class TestToneViolations:
    """Test detection of tone and style violations"""

    def test_detect_excessive_exclamation(self):
        detector = BrandGuidelinesDetector()
        text = "This is amazing!!! Buy now!!"
        findings = detector.analyze(text)

        tone = [f for f in findings if f.category == FindingCategory.TONE_VIOLATION]
        assert len(tone) >= 2  # Two instances of multiple exclamation marks

    def test_detect_all_caps(self):
        detector = BrandGuidelinesDetector()
        text = "This is URGENT and CRITICAL information."
        findings = detector.analyze(text)

        tone = [f for f in findings if f.category == FindingCategory.TONE_VIOLATION]
        # Should find URGENT and CRITICAL (≥4 caps)
        assert len(tone) >= 2

    def test_allow_acronyms(self):
        detector = BrandGuidelinesDetector()
        text = "Use our API to connect via HTTP or HTTPS with JSON data."
        findings = detector.analyze(text)

        tone = [f for f in findings if f.category == FindingCategory.TONE_VIOLATION]
        # API, HTTP, HTTPS, JSON should be in exceptions list
        assert len(tone) == 0

    def test_detect_excessive_question_marks(self):
        detector = BrandGuidelinesDetector()
        text = "What are you waiting for?? Sign up now???"
        findings = detector.analyze(text)

        tone = [f for f in findings if f.category == FindingCategory.TONE_VIOLATION]
        assert len(tone) >= 2

    def test_detect_informal_contractions(self):
        detector = BrandGuidelinesDetector()
        text = "You're gonna wanna check this out, ya know?"
        findings = detector.analyze(text)

        tone = [f for f in findings if f.category == FindingCategory.TONE_VIOLATION]
        # Should find: gonna, wanna, ya
        assert len(tone) >= 3

    def test_detect_excessive_ellipsis(self):
        detector = BrandGuidelinesDetector()
        text = "We offer many features.... and benefits...."
        findings = detector.analyze(text)

        tone = [f for f in findings if f.category == FindingCategory.TONE_VIOLATION]
        # Should find two instances of .... (4+ dots)
        assert len(tone) >= 2

    def test_single_exclamation_allowed(self):
        detector = BrandGuidelinesDetector()
        text = "We are excited to announce our new product!"
        findings = detector.analyze(text)

        tone = [f for f in findings if f.category == FindingCategory.TONE_VIOLATION]
        # Single exclamation mark is allowed
        assert len(tone) == 0


class TestComplianceScoring:
    """Test brand compliance scoring algorithm"""

    def test_perfect_compliance_score(self):
        detector = BrandGuidelinesDetector()
        text = "TechCorp's CloudSync provides a cost-effective solution for clients."
        findings = detector.analyze(text)
        metrics = calculate_brand_compliance_score(findings)

        assert metrics.compliance_score >= 0.95  # Nearly perfect
        assert metrics.total_violations == 0

    def test_low_compliance_score(self):
        detector = BrandGuidelinesDetector()
        text = "techcorp's cloudsync is cheap and free!!! Buy now to get unlimited storage guaranteed!!!"
        findings = detector.analyze(text)
        metrics = calculate_brand_compliance_score(findings)

        # Should have many violations
        assert metrics.compliance_score < 0.5  # Poor compliance
        assert metrics.total_violations > 5

    def test_scoring_weights(self):
        detector = BrandGuidelinesDetector()

        # Test capitalization penalty (0.10 per violation)
        text_cap = "techcorp and cloudsync"
        findings_cap = detector.analyze(text_cap)
        metrics_cap = calculate_brand_compliance_score(findings_cap)
        # 2 capitalization errors = 0.10 * 2 = 0.20 penalty → score = 0.80
        assert 0.75 <= metrics_cap.compliance_score <= 0.85

        # Test prohibited term penalty (0.15 per violation)
        text_prohibited = "This is cheap."
        findings_prohibited = detector.analyze(text_prohibited)
        metrics_prohibited = calculate_brand_compliance_score(findings_prohibited)
        # 1 prohibited term = 0.15 penalty → score = 0.85
        assert 0.80 <= metrics_prohibited.compliance_score <= 0.90

    def test_metrics_breakdown(self):
        detector = BrandGuidelinesDetector()
        text = "techcorp offers cheap services to customers. Login now!!!"
        findings = detector.analyze(text)
        metrics = calculate_brand_compliance_score(findings)

        assert metrics.capitalization_violations >= 1  # techcorp
        assert metrics.prohibited_term_count >= 1      # cheap
        assert metrics.non_preferred_term_count >= 2   # customers, Login
        assert metrics.tone_violations >= 1            # !!!
        assert metrics.total_violations == sum([
            metrics.capitalization_violations,
            metrics.prohibited_term_count,
            metrics.non_preferred_term_count,
            metrics.tone_violations
        ])


class TestConfigLoading:
    """Test brand config loading"""

    def test_load_default_config(self):
        detector = BrandGuidelinesDetector()
        assert detector.config.brand_name == "TechCorp"
        assert detector.config.version == "1.0.0"

    def test_config_has_patterns(self):
        detector = BrandGuidelinesDetector()
        assert len(detector.config.capitalization_rules) > 0
        assert len(detector.config.prohibited_terms) > 0
        assert len(detector.config.preferred_terminology) > 0
        assert len(detector.config.tone_rules) > 0

    def test_custom_config_path(self):
        # Test with explicit path
        config_path = Path(__file__).parent.parent.parent / 'bosspig' / 'config' / 'brand_config.json'
        detector = BrandGuidelinesDetector(config_path=config_path)
        assert detector.config.brand_name == "TechCorp"


class TestContextExtraction:
    """Test context and position tracking"""

    def test_line_number_tracking(self):
        detector = BrandGuidelinesDetector()
        text = "Line 1.\nLine 2 has techcorp.\nLine 3."
        findings = detector.analyze(text)

        cap_findings = [f for f in findings if f.category == FindingCategory.BRAND_CAPITALIZATION]
        assert len(cap_findings) >= 1
        assert cap_findings[0].line_number == 2

    def test_context_extraction(self):
        detector = BrandGuidelinesDetector()
        text = "We use techcorp's services daily."
        findings = detector.analyze(text)

        cap_findings = [f for f in findings if f.category == FindingCategory.BRAND_CAPITALIZATION]
        assert len(cap_findings) >= 1
        assert 'techcorp' in cap_findings[0].context.lower()

    def test_column_number_tracking(self):
        detector = BrandGuidelinesDetector()
        text = "We have techcorp here."
        findings = detector.analyze(text)

        cap_findings = [f for f in findings if f.category == FindingCategory.BRAND_CAPITALIZATION]
        assert len(cap_findings) >= 1
        assert cap_findings[0].column_number is not None
        assert cap_findings[0].column_number >= 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_text(self):
        detector = BrandGuidelinesDetector()
        findings = detector.analyze("")
        assert len(findings) == 0

    def test_whitespace_only(self):
        detector = BrandGuidelinesDetector()
        findings = detector.analyze("   \n   \n   ")
        assert len(findings) == 0

    def test_no_violations(self):
        detector = BrandGuidelinesDetector()
        text = "TechCorp provides excellent service."
        findings = detector.analyze(text)
        # Might have some findings (like "excellent" could be flagged), but shouldn't crash
        assert isinstance(findings, list)

    def test_very_long_text(self):
        detector = BrandGuidelinesDetector()
        # Create text with 100 instances of "cheap"
        text = " cheap" * 100
        findings = detector.analyze(text)

        prohibited = [f for f in findings if f.category == FindingCategory.PROHIBITED_TERM]
        assert len(prohibited) == 100

    def test_special_characters(self):
        detector = BrandGuidelinesDetector()
        text = "We have techcorp's @services# with $pricing%."
        findings = detector.analyze(text)

        cap_findings = [f for f in findings if f.category == FindingCategory.BRAND_CAPITALIZATION]
        assert len(cap_findings) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
