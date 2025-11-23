"""
BossPig Specificity Enforcement Detector

Detects vague, unmeasurable statements and enforces specific quantification.

Category 16: Specificity Enforcement
- Vague quantifiers detection
- Unmeasurable claims detection
- Relative statements without baselines

Author: Agent B (Sonnet) - Enhanced Categories
Week: 8 (Category Implementation)
Status: Week 8 Implementation (2025-11-22)
"""

from typing import List, Dict, Set, Optional
import re
from dataclasses import dataclass
from .core import Finding, Severity, FindingCategory


# ============================================================================
# VAGUE QUANTIFIER PATTERNS
# ============================================================================

VAGUE_QUANTIFIERS = {
    # Quantity vagueness
    r'\bseveral\b': "Specify exact number (e.g., 'three servers')",
    r'\bmany\b': "Specify quantity (e.g., '45 customers')",
    r'\bmost\b': "Specify percentage (e.g., '78% of users')",
    r'\bfew\b': "Specify exact number (e.g., 'two instances')",
    r'\ba lot of\b': "Specify quantity (e.g., '250 requests')",
    r'\bsome\b': "Specify number or percentage (e.g., 'some (15) files')",
    r'\bnumerous\b': "Specify exact count (e.g., '32 entries')",
    r'\bvarious\b': "List specific items or count (e.g., 'three types')",

    # Size/magnitude vagueness
    r'\bsignificant(?:ly)?\b': "Specify magnitude (e.g., '40% increase')",
    r'\bsubstantial(?:ly)?\b': "Quantify amount (e.g., '$500K budget')",
    r'\bconsiderable\b': "Provide specific value (e.g., '3.2TB storage')",
    r'\bsizeable\b': "Specify size (e.g., '2,500 sq ft')",
    r'\blarge\b': "Quantify size (e.g., 'large (50+ employee) team')",
    r'\bsmall\b': "Specify size (e.g., 'small (<10 user) deployment')",

    # Frequency vagueness
    r'\boften\b': "Specify frequency (e.g., 'daily' or '5 times/week')",
    r'\bfrequent(?:ly)?\b': "Provide frequency (e.g., 'every 2 hours')",
    r'\brare(?:ly)?\b': "Quantify occurrence (e.g., 'once per quarter')",
    r'\boccasional(?:ly)?\b': "Specify frequency (e.g., '2-3 times/month')",
    r'\bsometimes\b': "Provide specific conditions (e.g., 'during peak hours')",

    # Time vagueness
    r'\bsoon\b': "Specify date (e.g., 'by December 15')",
    r'\bshortly\b': "Provide timeframe (e.g., 'within 48 hours')",
    r'\brecent(?:ly)?\b': "Specify date (e.g., 'last week' or 'October 2025')",
    r'\bin the near future\b': "Provide specific date (e.g., 'Q1 2026')",
    r'\bquickly\b': "Quantify duration (e.g., 'within 5 minutes')",

    # Quality/degree vagueness
    r'\bfairly\b': "Quantify degree (e.g., 'fairly (75%) complete')",
    r'\brather\b': "Specify degree (e.g., 'rather high (>80%) confidence')",
    r'\bquite\b': "Quantify (e.g., 'quite significant (30%) improvement')",
    r'\bvery\b': "Replace with specific metric (e.g., 'very fast' → '<50ms')",
    r'\bextremely\b': "Provide specific measurement",
}

# ============================================================================
# UNMEASURABLE CLAIMS
# ============================================================================

UNMEASURABLE_PATTERNS = [
    # Future tense without metrics (look ahead up to 10 words for "by|to|from")
    (r'\b(?:will|shall) (?:improve|increase|enhance|grow|expand)\b(?!(?:\s+\w+){0,10}\s+(?:by|to|from)\s+\d)',
     "Add specific target (e.g., 'will increase by 25%')"),

    (r'\b(?:will|shall) (?:reduce|decrease|lower|minimize)\b(?!(?:\s+\w+){0,10}\s+(?:by|to|from)\s+\d)',
     "Add specific reduction target (e.g., 'will reduce by 30%')"),

    (r'\bexpect(?:ed)? to (?:see|achieve|reach)\b(?!\s+\d)',
     "Specify expected metric (e.g., 'expect to achieve 95% uptime')"),

    # Comparative without metrics
    (r'\b(?:better|worse|faster|slower|cheaper|more expensive)\b(?!\s+than\s+\S+)',
     "Add baseline comparison (e.g., 'faster than Q3 2025')"),

    (r'\bhigher|lower\b(?!\s+than\s+\S+)',
     "Provide baseline (e.g., 'higher than last quarter')"),

    # Qualitative claims
    (r'\b(?:world-class|best-in-class|industry-leading)\b',
     "Support with metrics or citations"),

    (r'\b(?:revolutionary|groundbreaking|game-changing)\b',
     "Quantify impact or provide evidence"),

    (r'\bunprecedented\b',
     "Provide historical comparison with data"),
]

# ============================================================================
# RELATIVE WITHOUT BASELINE
# ============================================================================

RELATIVE_PATTERNS = [
    (r'\b(?:faster|slower|quicker)\b(?!\s+than)',
     "Add baseline (e.g., 'faster than v2.0')"),

    (r'\b(?:cheaper|more expensive|costlier)\b(?!\s+than)',
     "Specify comparison point (e.g., 'cheaper than competitors')"),

    (r'\b(?:better|worse)\b(?!\s+than)',
     "Provide baseline (e.g., 'better than 2024 results')"),

    (r'\b(?:more|less) (?:efficient|effective|reliable)\b(?!\s+than)',
     "Add comparison baseline (e.g., 'more efficient than current system')"),

    (r'\b(?:improved|degraded)\b(?!\s+(?:from|since|compared))',
     "Specify baseline (e.g., 'improved from 85% to 92%')"),

    (r'\b(?:increased|decreased)\b(?!\s+(?:by|from|to))',
     "Quantify change (e.g., 'increased by 15%' or 'increased from 100 to 115')"),
]


# ============================================================================
# SPECIFICITY DETECTOR
# ============================================================================

@dataclass
class SpecificityMetrics:
    """Metrics for specificity analysis."""
    vague_quantifier_count: int
    unmeasurable_claim_count: int
    relative_without_baseline_count: int
    specificity_score: float  # 0.0 (very vague) to 1.0 (very specific)

    def to_dict(self) -> Dict:
        return {
            'vague_quantifiers': self.vague_quantifier_count,
            'unmeasurable_claims': self.unmeasurable_claim_count,
            'relative_without_baseline': self.relative_without_baseline_count,
            'specificity_score': self.specificity_score
        }


class SpecificityDetector:
    """
    Detects vague, unmeasurable language and enforces specific quantification.

    Example:
        detector = SpecificityDetector()
        findings = detector.analyze("We will significantly improve performance soon.")
        # Returns findings for "significantly" and "soon"
    """

    def __init__(self):
        self.vague_patterns = VAGUE_QUANTIFIERS
        self.unmeasurable_patterns = UNMEASURABLE_PATTERNS
        self.relative_patterns = RELATIVE_PATTERNS

    def analyze(self, text: str) -> List[Finding]:
        """
        Analyze text for specificity issues.

        Args:
            text: Text to analyze

        Returns:
            List of Finding objects for each specificity issue
        """
        findings = []

        # Detect vague quantifiers
        findings.extend(self._detect_vague_quantifiers(text))

        # Detect unmeasurable claims
        findings.extend(self._detect_unmeasurable_claims(text))

        # Detect relative statements without baselines
        findings.extend(self._detect_relative_without_baseline(text))

        return findings

    def _detect_vague_quantifiers(self, text: str) -> List[Finding]:
        """Detect vague quantifier words."""
        findings = []

        for pattern, suggestion in self.vague_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                findings.append(Finding(
                    category=FindingCategory.VAGUE_LANGUAGE,
                    severity=Severity.MEDIUM,
                    line_number=self._get_line_number(text, match.start()),
                    text=match.group(),
                    message=f"Vague quantifier: '{match.group()}'",
                    suggestion=suggestion,
                    column_number=self._get_column(text, match.start()),
                    context=self._get_context(text, match.start(), match.end())
                ))

        return findings

    def _detect_unmeasurable_claims(self, text: str) -> List[Finding]:
        """Detect claims that lack measurable targets."""
        findings = []

        for pattern, suggestion in self.unmeasurable_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                findings.append(Finding(
                    category=FindingCategory.UNMEASURABLE_CLAIM,
                    severity=Severity.HIGH,
                    line_number=self._get_line_number(text, match.start()),
                    text=match.group(),
                    message=f"Unmeasurable claim: '{match.group()}'",
                    suggestion=suggestion,
                    column_number=self._get_column(text, match.start()),
                    context=self._get_context(text, match.start(), match.end())
                ))

        return findings

    def _detect_relative_without_baseline(self, text: str) -> List[Finding]:
        """Detect relative statements without comparison baselines."""
        findings = []

        for pattern, suggestion in self.relative_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                findings.append(Finding(
                    category=FindingCategory.RELATIVE_WITHOUT_BASELINE,
                    severity=Severity.MEDIUM,
                    line_number=self._get_line_number(text, match.start()),
                    text=match.group(),
                    message=f"Relative statement without baseline: '{match.group()}'",
                    suggestion=suggestion,
                    column_number=self._get_column(text, match.start()),
                    context=self._get_context(text, match.start(), match.end())
                ))

        return findings

    def calculate_specificity_score(self, text: str, findings: List[Finding]) -> float:
        """
        Calculate overall specificity score (0.0 = very vague, 1.0 = very specific).

        Formula:
            base_score = 1.0
            penalty = (vague * 0.05) + (unmeasurable * 0.10) + (relative * 0.05)
            final_score = max(0.0, base_score - penalty)
        """
        vague_count = sum(1 for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE)
        unmeasurable_count = sum(1 for f in findings if f.category == FindingCategory.UNMEASURABLE_CLAIM)
        relative_count = sum(1 for f in findings if f.category == FindingCategory.RELATIVE_WITHOUT_BASELINE)

        base_score = 1.0
        penalty = (vague_count * 0.05) + (unmeasurable_count * 0.10) + (relative_count * 0.05)

        return max(0.0, base_score - penalty)

    def get_metrics(self, text: str) -> SpecificityMetrics:
        """Get comprehensive specificity metrics for text."""
        findings = self.analyze(text)

        vague_count = sum(1 for f in findings if f.category == FindingCategory.VAGUE_LANGUAGE)
        unmeasurable_count = sum(1 for f in findings if f.category == FindingCategory.UNMEASURABLE_CLAIM)
        relative_count = sum(1 for f in findings if f.category == FindingCategory.RELATIVE_WITHOUT_BASELINE)

        score = self.calculate_specificity_score(text, findings)

        return SpecificityMetrics(
            vague_quantifier_count=vague_count,
            unmeasurable_claim_count=unmeasurable_count,
            relative_without_baseline_count=relative_count,
            specificity_score=score
        )

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
# INTERACTIVE FIXER
# ============================================================================

class SpecificityFixer:
    """
    Interactive fixer for specificity issues.

    Since we need user input for actual values, this operates in INTERACTIVE mode.
    """

    def __init__(self):
        self.detector = SpecificityDetector()

    def fix_interactive(self, text: str, finding: Finding) -> str:
        """
        Fix a specificity issue with user input.

        In production, this would prompt the user via UI.
        For now, returns a template suggestion.
        """
        category = finding.category

        if category == FindingCategory.VAGUE_LANGUAGE:
            return self._fix_vague_quantifier(text, finding)
        elif category == FindingCategory.UNMEASURABLE_CLAIM:
            return self._fix_unmeasurable_claim(text, finding)
        elif category == FindingCategory.RELATIVE_WITHOUT_BASELINE:
            return self._fix_relative_statement(text, finding)

        return text

    def _fix_vague_quantifier(self, text: str, finding: Finding) -> str:
        """Generate fix template for vague quantifier."""
        # Extract the vague word
        match = re.search(r'\*\*(.+?)\*\*', finding.context)
        if match:
            vague_word = match.group(1)
            return f"[USER INPUT NEEDED: Replace '{vague_word}' with specific number/percentage]"
        return text

    def _fix_unmeasurable_claim(self, text: str, finding: Finding) -> str:
        """Generate fix template for unmeasurable claim."""
        return "[USER INPUT NEEDED: Add specific metric/target value]"

    def _fix_relative_statement(self, text: str, finding: Finding) -> str:
        """Generate fix template for relative statement."""
        return "[USER INPUT NEEDED: Specify comparison baseline]"
