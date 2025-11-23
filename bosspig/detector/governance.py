"""
BossPig Governance & Policy Tools Detector

Enforces organizational governance requirements including:
- Required sections in documents
- Required disclaimers
- Approval workflow validation
- Version control metadata
- Compliance framework adherence (HIPAA, SOC2, GDPR)

Category 18: Governance & Policy Tools
Author: Agent B (Sonnet) - Enhanced Categories
Week: 10 (Category Implementation)
Status: Week 10 Implementation (2025-11-22)
"""

from typing import List, Dict, Set, Optional
import re
import json
from pathlib import Path
from dataclasses import dataclass
from .core import Finding, Severity, FindingCategory


# ============================================================================
# GOVERNANCE CONFIG
# ============================================================================

@dataclass
class GovernanceConfig:
    """
    Governance configuration loaded from JSON.
    """
    document_type: str
    version: str
    required_sections: List[Dict]
    required_disclaimers: List[Dict]
    approval_workflows: List[Dict]
    version_control: List[Dict]
    compliance_frameworks: List[Dict]
    document_types: Dict

    @classmethod
    def from_json(cls, config_path: Path, document_type: str = "technical_documentation") -> 'GovernanceConfig':
        """Load governance config from JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get document type config
        doc_type_config = data.get('document_types', {}).get(document_type, {})

        return cls(
            document_type=document_type,
            version=data.get('version', '1.0.0'),
            required_sections=data.get('required_sections', {}).get('sections', []),
            required_disclaimers=data.get('required_disclaimers', {}).get('disclaimers', []),
            approval_workflows=data.get('approval_workflows', {}).get('workflows', []),
            version_control=data.get('version_control', {}).get('requirements', []),
            compliance_frameworks=data.get('compliance_frameworks', {}).get('frameworks', []),
            document_types=data.get('document_types', {})
        )


# ============================================================================
# GOVERNANCE DETECTOR
# ============================================================================

class GovernanceDetector:
    """
    Detects governance and policy violations in business documents.

    Example:
        detector = GovernanceDetector(document_type="healthcare")
        findings = detector.analyze(text)
        # Returns findings for missing required sections, disclaimers, etc.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        document_type: str = "technical_documentation"
    ):
        """
        Initialize governance detector.

        Args:
            config_path: Path to governance_config.json (default: bosspig/config/governance_config.json)
            document_type: Type of document to validate (technical_documentation, healthcare, data_policies, etc.)
        """
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent.parent / 'config' / 'governance_config.json'

        if not config_path.exists():
            raise FileNotFoundError(f"Governance config not found: {config_path}")

        self.config = GovernanceConfig.from_json(config_path, document_type)
        self.document_type = document_type

    def analyze(self, text: str) -> List[Finding]:
        """
        Analyze text for governance violations.

        Args:
            text: Text to analyze

        Returns:
            List of Finding objects for each violation
        """
        findings = []

        # Check required sections
        findings.extend(self._check_required_sections(text))

        # Check required disclaimers
        findings.extend(self._check_required_disclaimers(text))

        # Check approval workflows
        findings.extend(self._check_approval_workflows(text))

        # Check version control metadata
        findings.extend(self._check_version_control(text))

        # Check compliance frameworks (if applicable)
        findings.extend(self._check_compliance_frameworks(text))

        return findings

    def _check_required_sections(self, text: str) -> List[Finding]:
        """Check for required document sections."""
        findings = []

        for section in self.config.required_sections:
            if not section.get('required', True):
                continue  # Skip optional sections

            # Check if any of the section patterns match
            patterns = section.get('patterns', [])
            found = False

            for pattern in patterns:
                if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                    found = True
                    break

            if not found:
                # Section is missing
                severity_map = {
                    'critical': Severity.CRITICAL,
                    'high': Severity.HIGH,
                    'medium': Severity.MEDIUM,
                    'low': Severity.INFO
                }
                severity = severity_map.get(section.get('severity', 'medium'), Severity.MEDIUM)

                findings.append(Finding(
                    category=FindingCategory.MISSING_SECTION,
                    severity=severity,
                    line_number=1,  # Document-level finding
                    text="",
                    message=f"Missing required section: '{section['name']}'",
                    suggestion=f"Add a section header matching one of: {', '.join(patterns)}",
                    column_number=0,
                    context=f"Document type '{self.document_type}' requires this section"
                ))

        return findings

    def _check_required_disclaimers(self, text: str) -> List[Finding]:
        """Check for required disclaimers."""
        findings = []

        # Get required disclaimers for this document type
        doc_type_config = self.config.document_types.get(self.document_type, {})
        required_disclaimer_names = doc_type_config.get('required_disclaimers', [])

        for disclaimer in self.config.required_disclaimers:
            # Check if this disclaimer is required for this document type
            if disclaimer['name'] not in required_disclaimer_names:
                continue

            # Check if any of the disclaimer patterns match
            patterns = disclaimer.get('patterns', [])
            found = False

            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found = True
                    break

            if not found:
                # Disclaimer is missing
                severity_map = {
                    'critical': Severity.CRITICAL,
                    'high': Severity.HIGH,
                    'medium': Severity.MEDIUM,
                    'low': Severity.INFO
                }
                severity = severity_map.get(disclaimer.get('severity', 'high'), Severity.HIGH)

                location = disclaimer.get('location', 'any')
                location_msg = f" (should appear at {location})" if location != 'any' else ""

                findings.append(Finding(
                    category=FindingCategory.MISSING_DISCLAIMER,
                    severity=severity,
                    line_number=1,  # Document-level finding
                    text="",
                    message=f"Missing required disclaimer: '{disclaimer['name']}'{location_msg}",
                    suggestion=f"Add disclaimer containing one of: {', '.join(patterns)}",
                    column_number=0,
                    context=f"Required for document type '{self.document_type}'"
                ))

        return findings

    def _check_approval_workflows(self, text: str) -> List[Finding]:
        """Check for approval workflow metadata."""
        findings = []

        for workflow in self.config.approval_workflows:
            if not workflow.get('required', True):
                continue

            # Check if any of the workflow patterns match
            patterns = workflow.get('patterns', [])
            found = False

            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found = True
                    break

            if not found:
                # Approval metadata is missing
                severity_map = {
                    'critical': Severity.CRITICAL,
                    'high': Severity.HIGH,
                    'medium': Severity.MEDIUM,
                    'low': Severity.INFO
                }
                severity = severity_map.get(workflow.get('severity', 'medium'), Severity.MEDIUM)

                findings.append(Finding(
                    category=FindingCategory.MISSING_APPROVAL,
                    severity=severity,
                    line_number=1,  # Document-level finding
                    text="",
                    message=f"Missing approval metadata: '{workflow['name']}'",
                    suggestion=f"Add metadata matching one of: {', '.join(patterns)}",
                    column_number=0,
                    context="Approval workflow validation"
                ))

        return findings

    def _check_version_control(self, text: str) -> List[Finding]:
        """Check for version control metadata."""
        findings = []

        for requirement in self.config.version_control:
            if not requirement.get('required', True):
                continue

            # Check if any of the version control patterns match
            patterns = requirement.get('patterns', [])
            found = False

            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    found = True
                    break

            if not found:
                # Version control metadata is missing
                severity_map = {
                    'critical': Severity.CRITICAL,
                    'high': Severity.HIGH,
                    'medium': Severity.MEDIUM,
                    'low': Severity.INFO
                }
                severity = severity_map.get(requirement.get('severity', 'medium'), Severity.MEDIUM)

                findings.append(Finding(
                    category=FindingCategory.MISSING_VERSION_CONTROL,
                    severity=severity,
                    line_number=1,  # Document-level finding
                    text="",
                    message=f"Missing version control metadata: '{requirement['name']}'",
                    suggestion=f"Add metadata matching one of: {', '.join(patterns)}",
                    column_number=0,
                    context="Version control validation"
                ))

        return findings

    def _check_compliance_frameworks(self, text: str) -> List[Finding]:
        """Check for compliance framework requirements."""
        findings = []

        # Get required compliance frameworks for this document type
        doc_type_config = self.config.document_types.get(self.document_type, {})
        required_frameworks = doc_type_config.get('compliance_frameworks', [])

        for framework in self.config.compliance_frameworks:
            # Check if this framework is required for this document type
            if framework['name'] not in required_frameworks:
                continue

            # Check each required element for this framework
            for element in framework.get('required_elements', []):
                patterns = element.get('patterns', [])
                found = False

                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        found = True
                        break

                if not found:
                    # Compliance element is missing
                    severity_map = {
                        'critical': Severity.CRITICAL,
                        'high': Severity.HIGH,
                        'medium': Severity.MEDIUM,
                        'low': Severity.INFO
                    }
                    severity = severity_map.get(element.get('severity', 'high'), Severity.HIGH)

                    findings.append(Finding(
                        category=FindingCategory.COMPLIANCE_VIOLATION,
                        severity=severity,
                        line_number=1,  # Document-level finding
                        text="",
                        message=f"{framework['name']} compliance: Missing '{element['element']}'",
                        suggestion=f"Add content addressing: {', '.join(patterns)}",
                        column_number=0,
                        context=f"{framework['name']} framework requirement"
                    ))

        return findings


# ============================================================================
# GOVERNANCE METRICS
# ============================================================================

@dataclass
class GovernanceComplianceMetrics:
    """Metrics for governance compliance."""
    total_violations: int
    missing_sections: int
    missing_disclaimers: int
    missing_approvals: int
    missing_version_control: int
    compliance_violations: int
    governance_score: float  # 0.0 (many violations) to 1.0 (fully compliant)

    def to_dict(self) -> Dict:
        return {
            'total_violations': self.total_violations,
            'missing_sections': self.missing_sections,
            'missing_disclaimers': self.missing_disclaimers,
            'missing_approvals': self.missing_approvals,
            'missing_version_control': self.missing_version_control,
            'compliance_violations': self.compliance_violations,
            'governance_score': self.governance_score
        }


def calculate_governance_score(findings: List[Finding]) -> GovernanceComplianceMetrics:
    """
    Calculate governance compliance metrics.

    Scoring:
        base_score = 1.0
        penalty = (sections × 0.15) + (disclaimers × 0.20) + (approvals × 0.10) +
                  (version_control × 0.08) + (compliance × 0.25)
        final_score = max(0.0, base_score - penalty)
    """
    missing_sections = sum(1 for f in findings if f.category == FindingCategory.MISSING_SECTION)
    missing_disclaimers = sum(1 for f in findings if f.category == FindingCategory.MISSING_DISCLAIMER)
    missing_approvals = sum(1 for f in findings if f.category == FindingCategory.MISSING_APPROVAL)
    missing_version_control = sum(1 for f in findings if f.category == FindingCategory.MISSING_VERSION_CONTROL)
    compliance_violations = sum(1 for f in findings if f.category == FindingCategory.COMPLIANCE_VIOLATION)

    base_score = 1.0
    penalty = (missing_sections * 0.15) + (missing_disclaimers * 0.20) + \
              (missing_approvals * 0.10) + (missing_version_control * 0.08) + \
              (compliance_violations * 0.25)

    governance_score = max(0.0, base_score - penalty)

    return GovernanceComplianceMetrics(
        total_violations=len(findings),
        missing_sections=missing_sections,
        missing_disclaimers=missing_disclaimers,
        missing_approvals=missing_approvals,
        missing_version_control=missing_version_control,
        compliance_violations=compliance_violations,
        governance_score=governance_score
    )


if __name__ == "__main__":
    # Example usage
    detector = GovernanceDetector(document_type="technical_documentation")

    test_texts = [
        # Good example (complete)
        """# Purpose
        This document describes...

        # Scope
        Applies to all engineering teams.

        # Responsibilities
        The team lead is responsible for...

        # Version History
        v1.0.0 - 2025-11-22

        Confidential and proprietary information.

        Author: John Doe
        Reviewed by: Jane Smith
        Approval Date: 2025-11-22
        Status: Approved
        Version: 1.0.0
        Last Updated: 2025-11-22
        """,

        # Bad example (missing many elements)
        """Some content here without proper structure."""
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}")
        findings = detector.analyze(text)
        metrics = calculate_governance_score(findings)

        print(f"\nFindings: {len(findings)}")
        for finding in findings:
            print(f"  - [{finding.severity.value.upper()}] {finding.message}")

        print(f"\nGovernance Score: {metrics.governance_score:.2f}")
        print(f"  Missing Sections: {metrics.missing_sections}")
        print(f"  Missing Disclaimers: {metrics.missing_disclaimers}")
        print(f"  Missing Approvals: {metrics.missing_approvals}")
        print(f"  Missing Version Control: {metrics.missing_version_control}")
