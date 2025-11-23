"""
BossPig Brand Guidelines Compliance Detector

Enforces company-specific brand guidelines including:
- Brand name capitalization
- Prohibited terms
- Preferred terminology
- Tone and style rules

Category 17: Brand Guidelines Compliance
Author: Agent B (Sonnet) - Enhanced Categories
Week: 9 (Category Implementation)
Status: Week 9 Implementation (2025-11-22)
"""

from typing import List, Dict, Set, Optional
import re
import json
from pathlib import Path
from dataclasses import dataclass
from .core import Finding, Severity, FindingCategory


# ============================================================================
# BRAND GUIDELINES DETECTOR
# ============================================================================

@dataclass
class BrandConfig:
    """
    Brand guidelines configuration loaded from JSON.
    """
    brand_name: str
    version: str
    capitalization_rules: Dict
    prohibited_terms: List[Dict]
    preferred_terminology: List[Dict]
    tone_rules: List[Dict]

    @classmethod
    def from_json(cls, config_path: Path) -> 'BrandConfig':
        """Load brand config from JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(
            brand_name=data.get('brand_name', 'Unknown'),
            version=data.get('version', '1.0.0'),
            capitalization_rules=data.get('capitalization_rules', {}).get('patterns', {}),
            prohibited_terms=data.get('prohibited_terms', {}).get('terms', []),
            preferred_terminology=data.get('preferred_terminology', {}).get('preferences', []),
            tone_rules=data.get('tone_rules', {}).get('rules', [])
        )


class BrandGuidelinesDetector:
    """
    Detects brand guideline violations in business documents.

    Example:
        detector = BrandGuidelinesDetector()
        findings = detector.analyze("techcorp's cloudsync is cheap and free!")
        # Returns findings for: wrong capitalization, prohibited terms
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize brand guidelines detector.

        Args:
            config_path: Path to brand_config.json (default: bosspig/config/brand_config.json)
        """
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent.parent / 'config' / 'brand_config.json'

        if not config_path.exists():
            raise FileNotFoundError(f"Brand config not found: {config_path}")

        self.config = BrandConfig.from_json(config_path)
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        # Capitalization patterns
        self.capitalization_patterns = {}
        for brand, rule in self.config.capitalization_rules.items():
            incorrect_patterns = rule.get('incorrect_patterns', [])
            # Create alternation pattern for all incorrect forms
            # Note: Don't use re.IGNORECASE because we want exact matches of incorrect forms
            if incorrect_patterns:
                pattern = r'\b(' + '|'.join(re.escape(p) for p in incorrect_patterns) + r')\b'
                self.capitalization_patterns[brand] = {
                    'pattern': re.compile(pattern),  # Removed re.IGNORECASE
                    'correct': rule['correct'],
                    'severity': rule.get('severity', 'medium')
                }

        # Prohibited terms patterns
        self.prohibited_patterns = []
        for term in self.config.prohibited_terms:
            self.prohibited_patterns.append({
                'pattern': re.compile(term['pattern'], re.IGNORECASE),
                'reason': term['reason'],
                'severity': term.get('severity', 'medium'),
                'replacement': term.get('replacement', '')
            })

        # Preferred terminology patterns
        self.preferred_patterns = []
        for pref in self.config.preferred_terminology:
            self.preferred_patterns.append({
                'pattern': re.compile(pref['avoid'], re.IGNORECASE),
                'prefer': pref['prefer'],
                'reason': pref['reason'],
                'severity': pref.get('severity', 'medium')
            })

        # Tone rules patterns
        self.tone_patterns = []
        for rule in self.config.tone_rules:
            self.tone_patterns.append({
                'name': rule['name'],
                'pattern': re.compile(rule['pattern']),
                'message': rule['message'],
                'severity': rule.get('severity', 'medium'),
                'category': rule.get('category', 'tone'),
                'exceptions': rule.get('exceptions', [])
            })

    def analyze(self, text: str) -> List[Finding]:
        """
        Analyze text for brand guideline violations.

        Args:
            text: Text to analyze

        Returns:
            List of Finding objects for each violation
        """
        findings = []

        # Detect capitalization violations
        findings.extend(self._detect_capitalization_violations(text))

        # Detect prohibited terms
        findings.extend(self._detect_prohibited_terms(text))

        # Detect non-preferred terminology
        findings.extend(self._detect_preferred_terminology(text))

        # Detect tone violations
        findings.extend(self._detect_tone_violations(text))

        return findings

    def _detect_capitalization_violations(self, text: str) -> List[Finding]:
        """Detect incorrect brand name capitalization."""
        findings = []

        for brand, rule in self.capitalization_patterns.items():
            for match in rule['pattern'].finditer(text):
                # Map severity string to Severity enum
                severity_map = {
                    'critical': Severity.CRITICAL,
                    'high': Severity.HIGH,
                    'medium': Severity.MEDIUM,
                    'warning': Severity.WARNING,
                    'info': Severity.INFO
                }
                severity = severity_map.get(rule['severity'], Severity.MEDIUM)

                findings.append(Finding(
                    category=FindingCategory.BRAND_CAPITALIZATION,
                    severity=severity,
                    line_number=self._get_line_number(text, match.start()),
                    text=match.group(),
                    message=f"Incorrect capitalization: '{match.group()}'",
                    suggestion=f"Use correct brand capitalization: '{rule['correct']}'",
                    column_number=self._get_column(text, match.start()),
                    context=self._get_context(text, match.start(), match.end())
                ))

        return findings

    def _detect_prohibited_terms(self, text: str) -> List[Finding]:
        """Detect prohibited terms."""
        findings = []

        for rule in self.prohibited_patterns:
            for match in rule['pattern'].finditer(text):
                # Map severity string to Severity enum
                severity_map = {
                    'critical': Severity.CRITICAL,
                    'high': Severity.HIGH,
                    'medium': Severity.MEDIUM,
                    'warning': Severity.WARNING,
                    'info': Severity.INFO
                }
                severity = severity_map.get(rule['severity'], Severity.MEDIUM)

                suggestion = f"{rule['reason']}"
                if rule['replacement']:
                    suggestion += f" Suggested replacement: '{rule['replacement']}'"

                findings.append(Finding(
                    category=FindingCategory.PROHIBITED_TERM,
                    severity=severity,
                    line_number=self._get_line_number(text, match.start()),
                    text=match.group(),
                    message=f"Prohibited term: '{match.group()}'",
                    suggestion=suggestion,
                    column_number=self._get_column(text, match.start()),
                    context=self._get_context(text, match.start(), match.end())
                ))

        return findings

    def _detect_preferred_terminology(self, text: str) -> List[Finding]:
        """Detect non-preferred terminology."""
        findings = []

        for rule in self.preferred_patterns:
            for match in rule['pattern'].finditer(text):
                # Map severity string to Severity enum
                severity_map = {
                    'critical': Severity.CRITICAL,
                    'high': Severity.HIGH,
                    'medium': Severity.MEDIUM,
                    'warning': Severity.WARNING,
                    'low': Severity.INFO,  # Map 'low' to INFO
                    'info': Severity.INFO
                }
                severity = severity_map.get(rule['severity'], Severity.MEDIUM)

                findings.append(Finding(
                    category=FindingCategory.NON_PREFERRED_TERM,
                    severity=severity,
                    line_number=self._get_line_number(text, match.start()),
                    text=match.group(),
                    message=f"Non-preferred term: '{match.group()}'",
                    suggestion=f"Prefer '{rule['prefer']}'. {rule['reason']}",
                    column_number=self._get_column(text, match.start()),
                    context=self._get_context(text, match.start(), match.end())
                ))

        return findings

    def _detect_tone_violations(self, text: str) -> List[Finding]:
        """Detect tone and style violations."""
        findings = []

        for rule in self.tone_patterns:
            for match in rule['pattern'].finditer(text):
                # Check if this match is in the exceptions list
                if rule['exceptions'] and match.group() in rule['exceptions']:
                    continue

                # Map severity string to Severity enum
                severity_map = {
                    'critical': Severity.CRITICAL,
                    'high': Severity.HIGH,
                    'medium': Severity.MEDIUM,
                    'low': Severity.INFO,  # Map 'low' to INFO
                    'warning': Severity.WARNING,
                    'info': Severity.INFO
                }
                severity = severity_map.get(rule['severity'], Severity.MEDIUM)

                findings.append(Finding(
                    category=FindingCategory.TONE_VIOLATION,
                    severity=severity,
                    line_number=self._get_line_number(text, match.start()),
                    text=match.group(),
                    message=f"Tone violation ({rule['category']}): {rule['message']}",
                    suggestion=rule['message'],
                    column_number=self._get_column(text, match.start()),
                    context=self._get_context(text, match.start(), match.end())
                ))

        return findings

    def _get_line_number(self, text: str, pos: int) -> int:
        """Get line number for position in text."""
        return text[:pos].count('\n') + 1

    def _get_column(self, text: str, pos: int) -> int:
        """Get column number for position in text."""
        line_start = text.rfind('\n', 0, pos) + 1
        return pos - line_start

    def _get_context(self, text: str, start: int, end: int, context_chars: int = 50) -> str:
        """Get surrounding context for a match."""
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)

        before = text[context_start:start]
        match = text[start:end]
        after = text[end:context_end]

        return f"...{before}**{match}**{after}..."


# ============================================================================
# BRAND GUIDELINES METRICS
# ============================================================================

@dataclass
class BrandComplianceMetrics:
    """Metrics for brand guidelines compliance."""
    total_violations: int
    capitalization_violations: int
    prohibited_term_count: int
    non_preferred_term_count: int
    tone_violations: int
    compliance_score: float  # 0.0 (many violations) to 1.0 (perfect compliance)

    def to_dict(self) -> Dict:
        return {
            'total_violations': self.total_violations,
            'capitalization_violations': self.capitalization_violations,
            'prohibited_terms': self.prohibited_term_count,
            'non_preferred_terms': self.non_preferred_term_count,
            'tone_violations': self.tone_violations,
            'compliance_score': self.compliance_score
        }


def calculate_brand_compliance_score(findings: List[Finding]) -> BrandComplianceMetrics:
    """
    Calculate brand compliance metrics.

    Scoring:
        base_score = 1.0
        penalty = (capitalization × 0.10) + (prohibited × 0.15) + (non_preferred × 0.05) + (tone × 0.08)
        final_score = max(0.0, base_score - penalty)
    """
    capitalization_count = sum(1 for f in findings if f.category == FindingCategory.BRAND_CAPITALIZATION)
    prohibited_count = sum(1 for f in findings if f.category == FindingCategory.PROHIBITED_TERM)
    non_preferred_count = sum(1 for f in findings if f.category == FindingCategory.NON_PREFERRED_TERM)
    tone_count = sum(1 for f in findings if f.category == FindingCategory.TONE_VIOLATION)

    base_score = 1.0
    penalty = (capitalization_count * 0.10) + (prohibited_count * 0.15) + \
              (non_preferred_count * 0.05) + (tone_count * 0.08)

    compliance_score = max(0.0, base_score - penalty)

    return BrandComplianceMetrics(
        total_violations=len(findings),
        capitalization_violations=capitalization_count,
        prohibited_term_count=prohibited_count,
        non_preferred_term_count=non_preferred_count,
        tone_violations=tone_count,
        compliance_score=compliance_score
    )


if __name__ == "__main__":
    # Example usage
    detector = BrandGuidelinesDetector()

    test_texts = [
        # Good example
        "TechCorp's CloudSync service provides a cost-effective solution for clients.",

        # Bad example
        "techcorp's cloudsync is the #1 free service! Buy now!!!",

        # Mixed example
        "Login to github to purchase our cheap unlimited storage guaranteed."
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}: {text[:60]}...")
        findings = detector.analyze(text)
        metrics = calculate_brand_compliance_score(findings)

        print(f"\nFindings: {len(findings)}")
        for finding in findings:
            print(f"  - [{finding.severity.value.upper()}] {finding.message}")
            print(f"    Suggestion: {finding.suggestion}")

        print(f"\nCompliance Score: {metrics.compliance_score:.2f}")
        print(f"  Capitalization: {metrics.capitalization_violations}")
        print(f"  Prohibited Terms: {metrics.prohibited_term_count}")
        print(f"  Non-Preferred: {metrics.non_preferred_term_count}")
        print(f"  Tone Violations: {metrics.tone_violations}")
