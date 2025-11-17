"""
Peer Review Plugin for LMS Orchestration
Date: 2025-11-17
Version: 1.0.0

A comprehensive peer review system with:
- Blind/double-blind review
- Rubric-based assessment
- Calibration training
- Quality scoring
- Dispute resolution
- Knowledge graph integration
"""

from .plugin import PeerReviewPlugin
from .models import (
    PeerReviewAssignment,
    Submission,
    Review,
    Rubric,
    RubricCriterion,
    CalibrationScore,
    DisputeRequest
)
from .api import router

__all__ = [
    "PeerReviewPlugin",
    "PeerReviewAssignment",
    "Submission",
    "Review",
    "Rubric",
    "RubricCriterion",
    "CalibrationScore",
    "DisputeRequest",
    "router"
]
