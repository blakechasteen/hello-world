"""
Feature Detector Protocol — SAE Integration Point
===================================================

Unified interface for all detection methods in the alignment framework.
Current implementations use regex/keyword/Jaccard heuristics.
Future: SAE feature activation from Dark Trace.

Added: 2026-03-12 (Phase 6 Governance Wiring)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class DetectionResult:
    """Result from any FeatureDetector implementation."""
    score: float                          # 0.0-1.0 detection confidence
    activated_features: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class FeatureDetector(Protocol):
    """[SAE READY] Any detection method should conform to this interface.
    Current: regex/keyword/Jaccard implementations.
    Future: SAE feature activation from Dark Trace."""

    def detect(self, text: str, context: dict | None = None) -> DetectionResult:
        """Returns score (0.0-1.0) and activated features."""
        ...

    def get_active_features(self) -> list[str]:
        """Returns interpretable feature names that fired."""
        ...
