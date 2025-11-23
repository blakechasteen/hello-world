"""
Tests for BossPig Governance & Policy Tools Detector

Test Coverage:
- Required sections detection
- Required disclaimers checking
- Approval workflow validation
- Version control metadata
- Compliance framework adherence (HIPAA, SOC2, GDPR)

Created: 2025-11-22
Status: Week 10 Testing
"""

import pytest
from pathlib import Path
from bosspig.detector.governance import (
    GovernanceDetector,
    calculate_governance_score,
    GovernanceConfig
)
from bosspig.detector.core import FindingCategory, Severity


class TestRequiredSections:
    """Test detection of missing required sections"""

    def test_detect_missing_purpose(self):
        detector = GovernanceDetector(document_type="technical_documentation")
        text = "Some content without a Purpose section."
        findings = detector.analyze(text)

        missing_sections = [f for f in findings if f.category == FindingCategory.MISSING_SECTION]
        assert any('Purpose' in f.message for f in missing_sections)

    def test_detect_missing_scope(self):
        detector = GovernanceDetector(document_type="technical_documentation")
        text = "# Purpose\nSome purpose text."
        findings = detector.analyze(text)

        missing_sections = [f for f in findings if f.category == FindingCategory.MISSING_SECTION]
        assert any('Scope' in f.message for f in missing_sections)

    def test_all_required_sections_present(self):
        detector = GovernanceDetector(document_type="technical_documentation")
        text = """
        # Purpose
        This document describes...

        # Scope
        Applies to all teams.

        # Responsibilities
        Team lead is responsible.

        # Version History
        v1.0.0 - 2025-11-22
        """
        findings = detector.analyze(text)

        missing_sections = [f for f in findings if f.category == FindingCategory.MISSING_SECTION]
        # Should have no missing required sections
        assert len(missing_sections) == 0

    def test_alternative_section_names(self):
        detector = GovernanceDetector(document_type="technical_documentation")
        text = """
        # Overview
        (Purpose alternative)

        # Applicability
        (Scope alternative)

        # Roles
        (Responsibilities alternative)

        # Changelog
        (Version History alternative)
        """
        findings = detector.analyze(text)

        missing_sections = [f for f in findings if f.category == FindingCategory.MISSING_SECTION]
        # All required sections covered by alternatives
        assert len(missing_sections) == 0


class TestRequiredDisclaimers:
    """Test detection of missing required disclaimers"""

    def test_technical_doc_confidentiality_required(self):
        detector = GovernanceDetector(document_type="technical_documentation")
        text = "# Purpose\nSome content."
        findings = detector.analyze(text)

        missing_disclaimers = [f for f in findings if f.category == FindingCategory.MISSING_DISCLAIMER]
        assert any('Confidentiality Notice' in f.message for f in missing_disclaimers)

    def test_confidentiality_present(self):
        detector = GovernanceDetector(document_type="technical_documentation")
        text = """
        # Purpose
        Some content.

        Confidential and proprietary information.
        """
        findings = detector.analyze(text)

        missing_disclaimers = [f for f in findings if f.category == FindingCategory.MISSING_DISCLAIMER]
        # Confidentiality notice is present
        assert not any('Confidentiality Notice' in f.message for f in missing_disclaimers)

    def test_healthcare_hipaa_required(self):
        detector = GovernanceDetector(document_type="healthcare")
        text = "# Purpose\nHealthcare documentation."
        findings = detector.analyze(text)

        missing_disclaimers = [f for f in findings if f.category == FindingCategory.MISSING_DISCLAIMER]
        assert any('HIPAA Compliance' in f.message for f in missing_disclaimers)

    def test_hipaa_present(self):
        detector = GovernanceDetector(document_type="healthcare")
        text = """
        # Purpose
        Healthcare documentation.

        This system is HIPAA compliant and handles Protected Health Information (PHI).
        Confidential information.
        """
        findings = detector.analyze(text)

        missing_disclaimers = [f for f in findings if f.category == FindingCategory.MISSING_DISCLAIMER]
        # Both HIPAA and Confidentiality should be satisfied
        assert len(missing_disclaimers) == 0


class TestApprovalWorkflows:
    """Test detection of missing approval metadata"""

    def test_missing_author(self):
        detector = GovernanceDetector()
        text = "# Purpose\nSome content."
        findings = detector.analyze(text)

        missing_approvals = [f for f in findings if f.category == FindingCategory.MISSING_APPROVAL]
        assert any('Author Information' in f.message for f in missing_approvals)

    def test_missing_reviewer(self):
        detector = GovernanceDetector()
        text = """
        # Purpose
        Some content.

        Author: John Doe
        """
        findings = detector.analyze(text)

        missing_approvals = [f for f in findings if f.category == FindingCategory.MISSING_APPROVAL]
        assert any('Reviewer Approval' in f.message for f in missing_approvals)

    def test_missing_approval_date(self):
        detector = GovernanceDetector()
        text = """
        # Purpose
        Some content.

        Author: John Doe
        Reviewed by: Jane Smith
        """
        findings = detector.analyze(text)

        missing_approvals = [f for f in findings if f.category == FindingCategory.MISSING_APPROVAL]
        assert any('Approval Date' in f.message for f in missing_approvals)

    def test_all_approvals_present(self):
        detector = GovernanceDetector()
        text = """
        # Purpose
        Some content.

        Author: John Doe
        Reviewed by: Jane Smith
        Approval Date: 2025-11-22
        Status: Approved
        """
        findings = detector.analyze(text)

        missing_approvals = [f for f in findings if f.category == FindingCategory.MISSING_APPROVAL]
        # All approval metadata present
        assert len(missing_approvals) == 0


class TestVersionControl:
    """Test detection of missing version control metadata"""

    def test_missing_version_number(self):
        detector = GovernanceDetector()
        text = "# Purpose\nSome content."
        findings = detector.analyze(text)

        missing_vc = [f for f in findings if f.category == FindingCategory.MISSING_VERSION_CONTROL]
        assert any('Version Number' in f.message for f in missing_vc)

    def test_version_number_present(self):
        detector = GovernanceDetector()
        text = """
        # Purpose
        Some content.

        Version: 1.0.0
        """
        findings = detector.analyze(text)

        missing_vc = [f for f in findings if f.category == FindingCategory.MISSING_VERSION_CONTROL]
        # Version Number should be satisfied
        assert not any('Version Number' in f.message for f in missing_vc)

    def test_missing_last_updated(self):
        detector = GovernanceDetector()
        text = """
        # Purpose
        Some content.

        Version: 1.0.0
        """
        findings = detector.analyze(text)

        missing_vc = [f for f in findings if f.category == FindingCategory.MISSING_VERSION_CONTROL]
        assert any('Last Updated Date' in f.message for f in missing_vc)

    def test_all_version_control_present(self):
        detector = GovernanceDetector()
        text = """
        # Purpose
        Some content.

        Version: 1.0.0
        Last Updated: 2025-11-22
        """
        findings = detector.analyze(text)

        missing_vc = [f for f in findings if f.category == FindingCategory.MISSING_VERSION_CONTROL]
        # All required version control metadata present
        assert len(missing_vc) == 0


class TestComplianceFrameworks:
    """Test compliance framework validation (HIPAA, SOC2, GDPR)"""

    def test_hipaa_phi_handling_required(self):
        detector = GovernanceDetector(document_type="healthcare")
        text = """
        # Purpose
        Healthcare system.

        Confidential. HIPAA compliant.
        """
        findings = detector.analyze(text)

        compliance = [f for f in findings if f.category == FindingCategory.COMPLIANCE_VIOLATION]
        # Should require PHI handling, access controls, audit trail, encryption
        assert len(compliance) >= 1  # At least one element missing

    def test_hipaa_all_elements_present(self):
        detector = GovernanceDetector(document_type="healthcare")
        text = """
        # Purpose
        Healthcare system documentation.

        # Scope
        Handles Protected Health Information (PHI).

        # Security
        Access control and authentication required.
        Audit log maintained for all PHI access.
        All data encrypted with TLS/SSL.

        Confidential. HIPAA compliant.
        """
        findings = detector.analyze(text)

        compliance = [f for f in findings if f.category == FindingCategory.COMPLIANCE_VIOLATION]
        # All HIPAA elements present
        assert len(compliance) == 0

    def test_soc2_required_for_data_policies(self):
        detector = GovernanceDetector(document_type="data_policies")
        text = """
        # Purpose
        Data retention policy.

        Data retention period is 7 years.
        """
        findings = detector.analyze(text)

        compliance = [f for f in findings if f.category == FindingCategory.COMPLIANCE_VIOLATION]
        # Should require SOC2 elements
        assert len(compliance) >= 1

    def test_gdpr_required_for_data_policies(self):
        detector = GovernanceDetector(document_type="data_policies")
        text = """
        # Purpose
        Data retention policy.

        Data retention period is 7 years.
        Security controls in place.
        """
        findings = detector.analyze(text)

        compliance = [f for f in findings if f.category == FindingCategory.COMPLIANCE_VIOLATION]
        # Should require GDPR elements (data subject rights, lawful basis, etc.)
        assert any('GDPR' in f.message for f in compliance)

    def test_no_compliance_for_technical_doc(self):
        detector = GovernanceDetector(document_type="technical_documentation")
        text = """
        # Purpose
        Technical documentation.

        # Scope
        Applies to engineering.

        # Responsibilities
        Team lead.

        # Version History
        v1.0.0

        Confidential.
        Author: John
        Reviewed by: Jane
        Approval Date: 2025-11-22
        Status: Approved
        Version: 1.0.0
        Last Updated: 2025-11-22
        """
        findings = detector.analyze(text)

        compliance = [f for f in findings if f.category == FindingCategory.COMPLIANCE_VIOLATION]
        # No compliance frameworks required for technical_documentation
        assert len(compliance) == 0


class TestGovernanceScoring:
    """Test governance compliance scoring"""

    def test_perfect_compliance_score(self):
        detector = GovernanceDetector(document_type="technical_documentation")
        text = """
        # Purpose
        Technical documentation.

        # Scope
        Applies to engineering.

        # Responsibilities
        Team lead responsible.

        # Version History
        v1.0.0 - 2025-11-22

        Confidential and proprietary information.

        Author: John Doe
        Reviewed by: Jane Smith
        Approval Date: 2025-11-22
        Status: Approved

        Version: 1.0.0
        Last Updated: 2025-11-22
        """
        findings = detector.analyze(text)
        metrics = calculate_governance_score(findings)

        assert metrics.governance_score >= 0.95  # Nearly perfect
        assert metrics.total_violations == 0

    def test_low_compliance_score(self):
        detector = GovernanceDetector(document_type="technical_documentation")
        text = "Just some random content without any structure."
        findings = detector.analyze(text)
        metrics = calculate_governance_score(findings)

        # Missing: sections (4), disclaimer (1), approvals (4), version control (2) = 11 violations
        assert metrics.governance_score < 0.2  # Very poor compliance
        assert metrics.total_violations >= 10

    def test_scoring_weights(self):
        detector = GovernanceDetector(document_type="healthcare")

        # Test with missing compliance elements (highest penalty: 0.25)
        text1 = """
        # Purpose
        Healthcare.

        # Scope
        All patients.

        # Responsibilities
        Team.

        Confidential. HIPAA compliant.
        Author: John
        Reviewed by: Jane
        Approval Date: 2025-11-22
        Status: Approved
        Version: 1.0
        Last Updated: 2025-11-22
        """
        findings1 = detector.analyze(text1)
        metrics1 = calculate_governance_score(findings1)

        # Should have compliance violations (HIPAA elements missing)
        # 4 HIPAA elements × 0.25 = 1.00 penalty (capped at 0.0)
        assert metrics1.compliance_violations >= 3
        assert metrics1.governance_score < 0.3

    def test_metrics_breakdown(self):
        detector = GovernanceDetector(document_type="technical_documentation")
        text = """
        # Purpose
        Some doc.

        Random content.
        """
        findings = detector.analyze(text)
        metrics = calculate_governance_score(findings)

        # Should be missing: Scope, Responsibilities, Version History (3 sections),
        # Confidentiality disclaimer (1), approvals (4), version control (2)
        assert metrics.missing_sections >= 3
        assert metrics.missing_disclaimers >= 1
        assert metrics.missing_approvals >= 4
        assert metrics.missing_version_control >= 2


class TestConfigLoading:
    """Test governance config loading"""

    def test_load_default_config(self):
        detector = GovernanceDetector()
        assert detector.config.document_type == "technical_documentation"
        assert detector.config.version == "1.0.0"

    def test_config_has_requirements(self):
        detector = GovernanceDetector()
        assert len(detector.config.required_sections) > 0
        assert len(detector.config.approval_workflows) > 0
        assert len(detector.config.version_control) > 0

    def test_custom_document_type(self):
        detector = GovernanceDetector(document_type="healthcare")
        assert detector.document_type == "healthcare"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_text(self):
        detector = GovernanceDetector()
        findings = detector.analyze("")
        # Should find many missing elements
        assert len(findings) >= 10

    def test_whitespace_only(self):
        detector = GovernanceDetector()
        findings = detector.analyze("   \n   \n   ")
        # Should find many missing elements
        assert len(findings) >= 10

    def test_case_insensitive_section_detection(self):
        detector = GovernanceDetector()
        text_lower = "# purpose\nSome text."
        text_upper = "# PURPOSE\nSome text."

        findings_lower = detector.analyze(text_lower)
        findings_upper = detector.analyze(text_upper)

        missing_lower = [f for f in findings_lower if f.category == FindingCategory.MISSING_SECTION and 'Purpose' in f.message]
        missing_upper = [f for f in findings_upper if f.category == FindingCategory.MISSING_SECTION and 'Purpose' in f.message]

        # Both should recognize Purpose section (case-insensitive)
        assert len(missing_lower) == 0
        assert len(missing_upper) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
