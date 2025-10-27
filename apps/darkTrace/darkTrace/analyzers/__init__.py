"""
darkTrace Analyzers - Layer 2: Analysis
========================================
Trajectory prediction and pattern recognition.

This layer provides:
- TrajectoryPredictor: System identification via differential equations
- FingerprintGenerator: Create unique semantic fingerprints
- PatternRecognizer: Detect recurring patterns in trajectories
- AttractorDetector: Find stable semantic attractors
"""

from darkTrace.analyzers.trajectory_predictor import TrajectoryPredictor, PredictedState
from darkTrace.analyzers.fingerprint_generator import FingerprintGenerator, SemanticFingerprint
from darkTrace.analyzers.pattern_recognizer import PatternRecognizer, Pattern
from darkTrace.analyzers.attractor_detector import AttractorDetector, Attractor

__all__ = [
    "TrajectoryPredictor",
    "PredictedState",
    "FingerprintGenerator",
    "SemanticFingerprint",
    "PatternRecognizer",
    "Pattern",
    "AttractorDetector",
    "Attractor",
]
