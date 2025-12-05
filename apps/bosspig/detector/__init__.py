"""
BossPig Business Documentation AI Slop Detector

Detects 15 categories of quality issues in business writing:
1. Corporate jargon
2. Vague commitments
3. Missing dates/deadlines
4. AI hallucination markers
5. Passive voice overload
6. Redundant phrasing
7. Weasel words
8. Inconsistent formatting
9. Empty headers
10. Data quality issues
11. Compliance red flags
12. Meeting notes anti-patterns
13. Email anti-patterns
14. Meaningless metrics
15. Unclear ownership

Usage:
    from bosspig.detector import BossPigDetector

    detector = BossPigDetector()
    findings = detector.analyze("proposal.docx")
    print(f"Quality Score: {findings.quality_score}/100")

Author: Agent B (Sonnet)
Week: 1-3 (Top 5 categories) - COMPLETE
Status: Week 1-3 Complete (2025-11-22)
"""

from .detector import BossPigDetector
from .scorer import QualityScorer
from .core import (
    Finding,
    BossPigFindings,
    QualityMetrics,
    FindingCategory,
    Severity
)

__all__ = [
    "BossPigDetector",
    "QualityScorer",
    "Finding",
    "BossPigFindings",
    "QualityMetrics",
    "FindingCategory",
    "Severity"
]
