"""
BossPig: Business Documentation AI Slop Detector

Detects corporate jargon, vague commitments, missing deadlines,
AI hallucinations, and unclear ownership in business writing.

Created: 2025-11-22
Version: 0.1.0 (MVP - Week 1-3)
"""

from .detector.detector import BossPigDetector
from .detector.scorer import QualityScorer
from .detector.core import (
    Finding,
    BossPigFindings,
    QualityMetrics,
    FindingCategory,
    Severity
)

__version__ = "0.1.0"
__all__ = [
    "BossPigDetector",
    "QualityScorer",
    "Finding",
    "BossPigFindings",
    "QualityMetrics",
    "FindingCategory",
    "Severity"
]
