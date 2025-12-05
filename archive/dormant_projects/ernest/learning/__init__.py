"""
Ernest Learning Module
======================
Adaptive learning for creative writing refinement.

Main exports:
- ErnestLearningEngine: Complete learning system
- WritingModePrefs: Mode preference tracking
- RefinementPassStats: Pass effectiveness tracking
- refine_with_ernest: Quick refinement helper
- analyze_with_ernest: Quick analysis helper
"""

from .ernest_learning_engine import (
    ErnestLearningEngine,
    WritingModePrefs,
    RefinementPassStats,
    CreativePatterns,
    ErnestLearningMetrics,
    refine_with_ernest,
    analyze_with_ernest
)

__all__ = [
    "ErnestLearningEngine",
    "WritingModePrefs",
    "RefinementPassStats",
    "CreativePatterns",
    "ErnestLearningMetrics",
    "refine_with_ernest",
    "analyze_with_ernest"
]
