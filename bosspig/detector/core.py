"""
BossPig Core Detection Engine

Detection algorithm infrastructure with pattern matching, severity classification,
and finding data structures.

Created: 2025-11-22
Status: Production Ready
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set
import re
from pathlib import Path


# Severity levels for findings
class Severity(Enum):
    """Severity classification for detected issues"""
    CRITICAL = "critical"  # Major issues that must be fixed
    HIGH = "high"          # High priority issues
    MEDIUM = "medium"      # Medium priority issues
    WARNING = "warning"    # Issues that should be fixed
    INFO = "info"          # Minor issues, suggestions


# Finding categories
class FindingCategory(Enum):
    """Categories of business slop detection"""
    CORPORATE_JARGON = "corporate_jargon"
    VAGUE_COMMITMENTS = "vague_commitments"
    MEANINGLESS_METRICS = "meaningless_metrics"
    PASSIVE_VOICE = "passive_voice"
    MISSING_DATES = "missing_dates"
    UNCLEAR_OWNERSHIP = "unclear_ownership"
    AI_HALLUCINATION = "ai_hallucination"
    REDUNDANT_PHRASES = "redundant_phrases"
    WEASEL_WORDS = "weasel_words"
    FORMATTING_ISSUES = "formatting_issues"
    EMPTY_SECTIONS = "empty_sections"
    DATA_QUALITY = "data_quality"
    COMPLIANCE_ISSUES = "compliance_issues"
    MEETING_NOTES_ANTIPATTERNS = "meeting_notes_antipatterns"
    EMAIL_ANTIPATTERNS = "email_antipatterns"
    # Category 16: Specificity Enforcement
    VAGUE_LANGUAGE = "vague_language"
    UNMEASURABLE_CLAIM = "unmeasurable_claim"
    RELATIVE_WITHOUT_BASELINE = "relative_without_baseline"
    # Category 17: Brand Guidelines Compliance
    BRAND_CAPITALIZATION = "brand_capitalization"
    PROHIBITED_TERM = "prohibited_term"
    NON_PREFERRED_TERM = "non_preferred_term"
    TONE_VIOLATION = "tone_violation"
    # Category 18: Governance & Policy Tools
    MISSING_SECTION = "missing_section"
    MISSING_DISCLAIMER = "missing_disclaimer"
    MISSING_APPROVAL = "missing_approval"
    MISSING_VERSION_CONTROL = "missing_version_control"
    COMPLIANCE_VIOLATION = "compliance_violation"


@dataclass
class Finding:
    """
    Represents a single detected issue in a document.

    Attributes:
        category: Type of issue (jargon, vague commitment, etc.)
        severity: Critical/Warning/Info
        line_number: Line number where issue was found
        column_number: Column number (if available)
        text: The problematic text snippet
        message: Human-readable explanation of the issue
        suggestion: Recommended fix
        context: Surrounding text for context (optional)
    """
    category: FindingCategory
    severity: Severity
    line_number: int
    text: str
    message: str
    suggestion: str
    column_number: Optional[int] = None
    context: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert finding to dictionary for JSON export"""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "text": self.text,
            "message": self.message,
            "suggestion": self.suggestion,
            "context": self.context
        }


@dataclass
class QualityMetrics:
    """
    Quality metrics for a document.

    Scores range from 0.0 to 1.0, with 1.0 being perfect.
    """
    clarity_score: float = 0.0       # 1.0 - (jargon_count / total_words)
    specificity_score: float = 0.0   # 1.0 - (vague_phrases / total_sentences)
    actionability_score: float = 0.0 # action_items / total_paragraphs
    professionalism_score: float = 0.0  # 1.0 - (hallucinations + formatting_issues) / total_sections
    completeness_score: float = 0.0 # filled_sections / total_sections

    def overall_score(self) -> int:
        """
        Calculate overall quality score (0-100).

        Weighted average:
        - Clarity: 30%
        - Specificity: 25%
        - Actionability: 20%
        - Professionalism: 15%
        - Completeness: 10%
        """
        score = (
            0.30 * self.clarity_score +
            0.25 * self.specificity_score +
            0.20 * self.actionability_score +
            0.15 * self.professionalism_score +
            0.10 * self.completeness_score
        )
        return round(score * 100)

    def grade(self) -> str:
        """
        Convert score to letter grade.

        90-100: A (Excellent)
        80-89: B (Good)
        70-79: C (Fair)
        60-69: D (Poor)
        0-59: F (Critical)
        """
        score = self.overall_score()
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def grade_label(self) -> str:
        """Get descriptive label for grade"""
        labels = {
            "A": "Excellent",
            "B": "Good",
            "C": "Fair",
            "D": "Poor",
            "F": "Critical"
        }
        return labels.get(self.grade(), "Unknown")


@dataclass
class BossPigFindings:
    """
    Complete analysis results for a document.

    Attributes:
        findings: List of all detected issues
        quality_metrics: Quality scores
        document_stats: Document statistics (word count, sentences, etc.)
    """
    findings: List[Finding] = field(default_factory=list)
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    document_stats: Dict = field(default_factory=dict)

    @property
    def quality_score(self) -> int:
        """Overall quality score (0-100)"""
        return self.quality_metrics.overall_score()

    @property
    def grade(self) -> str:
        """Letter grade (A-F)"""
        return self.quality_metrics.grade()

    @property
    def grade_label(self) -> str:
        """Descriptive grade label"""
        return self.quality_metrics.grade_label()

    def findings_by_severity(self, severity: Severity) -> List[Finding]:
        """Get all findings with specific severity"""
        return [f for f in self.findings if f.severity == severity]

    def findings_by_category(self, category: FindingCategory) -> List[Finding]:
        """Get all findings of specific category"""
        return [f for f in self.findings if f.category == category]

    def critical_count(self) -> int:
        """Count of critical findings"""
        return len(self.findings_by_severity(Severity.CRITICAL))

    def warning_count(self) -> int:
        """Count of warning findings"""
        return len(self.findings_by_severity(Severity.WARNING))

    def info_count(self) -> int:
        """Count of info findings"""
        return len(self.findings_by_severity(Severity.INFO))

    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = []
        lines.append(f"BossPig Analysis Results")
        lines.append(f"========================")
        lines.append(f"Quality Score: {self.quality_score}/100 ({self.grade} - {self.grade_label})")
        lines.append(f"")
        lines.append(f"Findings:")
        lines.append(f"  Critical: {self.critical_count()}")
        lines.append(f"  Warnings: {self.warning_count()}")
        lines.append(f"  Info: {self.info_count()}")
        lines.append(f"")
        lines.append(f"Quality Breakdown:")
        lines.append(f"  Clarity: {int(self.quality_metrics.clarity_score * 100)}/100")
        lines.append(f"  Specificity: {int(self.quality_metrics.specificity_score * 100)}/100")
        lines.append(f"  Actionability: {int(self.quality_metrics.actionability_score * 100)}/100")
        lines.append(f"  Professionalism: {int(self.quality_metrics.professionalism_score * 100)}/100")
        lines.append(f"  Completeness: {int(self.quality_metrics.completeness_score * 100)}/100")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert findings to dictionary for JSON export"""
        return {
            "quality_score": self.quality_score,
            "grade": self.grade,
            "grade_label": self.grade_label,
            "metrics": {
                "clarity": self.quality_metrics.clarity_score,
                "specificity": self.quality_metrics.specificity_score,
                "actionability": self.quality_metrics.actionability_score,
                "professionalism": self.quality_metrics.professionalism_score,
                "completeness": self.quality_metrics.completeness_score
            },
            "findings": {
                "critical": self.critical_count(),
                "warnings": self.warning_count(),
                "info": self.info_count(),
                "total": len(self.findings),
                "items": [f.to_dict() for f in self.findings]
            },
            "document_stats": self.document_stats
        }


# Pattern matching infrastructure
class PatternMatcher:
    """
    Infrastructure for pattern matching using regex and NLP.

    Supports:
    - Regex pattern matching
    - Case-insensitive matching
    - Context extraction
    - Line and column number tracking
    """

    def __init__(self, case_sensitive: bool = False):
        """
        Initialize pattern matcher.

        Args:
            case_sensitive: Whether to perform case-sensitive matching
        """
        self.case_sensitive = case_sensitive
        self.compiled_patterns: Dict[str, re.Pattern] = {}

    def compile_pattern(self, pattern: str, flags: int = 0) -> re.Pattern:
        """
        Compile regex pattern with optional flags.

        Args:
            pattern: Regex pattern string
            flags: Optional regex flags

        Returns:
            Compiled regex pattern
        """
        if not self.case_sensitive:
            flags |= re.IGNORECASE

        key = f"{pattern}:{flags}"
        if key not in self.compiled_patterns:
            self.compiled_patterns[key] = re.compile(pattern, flags)

        return self.compiled_patterns[key]

    def find_all_matches(
        self,
        text: str,
        pattern: str,
        context_chars: int = 50
    ) -> List[Dict]:
        """
        Find all matches of pattern in text with context.

        Args:
            text: Text to search
            pattern: Regex pattern
            context_chars: Number of chars before/after match for context

        Returns:
            List of match dictionaries with position, matched text, and context
        """
        regex = self.compile_pattern(pattern)
        matches = []

        for match in regex.finditer(text):
            start = match.start()
            end = match.end()

            # Calculate line and column number
            line_num = text[:start].count('\n') + 1
            col_num = start - text[:start].rfind('\n') - 1

            # Extract context
            context_start = max(0, start - context_chars)
            context_end = min(len(text), end + context_chars)
            context = text[context_start:context_end]

            matches.append({
                "text": match.group(),
                "start": start,
                "end": end,
                "line_number": line_num,
                "column_number": col_num,
                "context": context
            })

        return matches

    def contains_pattern(self, text: str, pattern: str) -> bool:
        """Check if text contains pattern"""
        regex = self.compile_pattern(pattern)
        return regex.search(text) is not None


# Document statistics calculator
class DocumentStats:
    """Calculate document statistics for quality scoring"""

    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text"""
        return len(re.findall(r'\b\w+\b', text))

    @staticmethod
    def count_sentences(text: str) -> int:
        """Count sentences in text (rough heuristic)"""
        # Count periods, exclamation marks, question marks
        return len(re.findall(r'[.!?]+', text))

    @staticmethod
    def count_paragraphs(text: str) -> int:
        """Count paragraphs (double newline separated)"""
        paragraphs = re.split(r'\n\s*\n', text.strip())
        return len([p for p in paragraphs if p.strip()])

    @staticmethod
    def count_sections(text: str) -> int:
        """Count sections (markdown headers or blank-line separated)"""
        # Count markdown headers
        headers = len(re.findall(r'^#{1,6}\s+.+$', text, re.MULTILINE))
        if headers > 0:
            return headers
        # Fallback to paragraph count
        return DocumentStats.count_paragraphs(text)

    @staticmethod
    def calculate_stats(text: str) -> Dict:
        """Calculate all document statistics"""
        return {
            "word_count": DocumentStats.count_words(text),
            "sentence_count": DocumentStats.count_sentences(text),
            "paragraph_count": DocumentStats.count_paragraphs(text),
            "section_count": DocumentStats.count_sections(text),
            "character_count": len(text),
            "line_count": text.count('\n') + 1
        }


if __name__ == "__main__":
    # Test pattern matching
    matcher = PatternMatcher()

    test_text = """
    We need to synergize our core competencies to leverage the low-hanging fruit.
    Let's circle back on this next week. We will try to finish this soon.
    """

    matches = matcher.find_all_matches(test_text, r'\bsynergize\b')
    print(f"Found {len(matches)} matches for 'synergize':")
    for match in matches:
        print(f"  Line {match['line_number']}: '{match['text']}'")

    # Test document stats
    stats = DocumentStats.calculate_stats(test_text)
    print(f"\nDocument Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test quality metrics
    metrics = QualityMetrics(
        clarity_score=0.75,
        specificity_score=0.60,
        actionability_score=0.50,
        professionalism_score=0.80,
        completeness_score=0.90
    )
    print(f"\nQuality Score: {metrics.overall_score()}/100")
    print(f"Grade: {metrics.grade()} - {metrics.grade_label()}")
