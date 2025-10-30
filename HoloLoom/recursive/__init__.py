"""
Recursive Learning - HoloLoom + Scratchpad + Loop Engine Integration
======================================================================
Phase 1: Scratchpad Integration for Full Provenance Tracking

This module connects HoloLoom's WeavingOrchestrator with Promptly's
Scratchpad to enable full provenance tracking and recursive refinement.

Key Components:
- ScratchpadOrchestrator: HoloLoom + Scratchpad integration
- ProvenanceTracker: Extract provenance from Spacetime traces
- RecursiveRefiner: Trigger refinement loops on low confidence

Vision: Self-improving knowledge system that learns from usage patterns.
"""

from .scratchpad_integration import (
    ScratchpadOrchestrator,
    ScratchpadConfig,
    ProvenanceTracker,
    RecursiveRefiner,
    weave_with_scratchpad,
)

from .loop_integration import (
    LearningLoopEngine,
    LearningLoopConfig,
    PatternExtractor,
    PatternLearner,
    LearnedPattern,
    weave_with_learning,
)

from .hot_patterns import (
    HotPatternFeedbackEngine,
    HotPatternConfig,
    HotPatternTracker,
    AdaptiveRetriever,
    UsageRecord,
)

from .advanced_refinement import (
    AdvancedRefiner,
    RefinementStrategy,
    RefinementResult,
    QualityMetrics,
    RefinementPattern,
    refine_with_strategy,
)

from .full_learning_loop import (
    FullLearningEngine,
    ThompsonPriors,
    PolicyWeights,
    BackgroundLearner,
    LearningMetrics,
    weave_with_full_learning,
)

__all__ = [
    # Phase 1: Scratchpad Integration
    "ScratchpadOrchestrator",
    "ScratchpadConfig",
    "ProvenanceTracker",
    "RecursiveRefiner",
    "weave_with_scratchpad",

    # Phase 2: Loop Engine Integration
    "LearningLoopEngine",
    "LearningLoopConfig",
    "PatternExtractor",
    "PatternLearner",
    "LearnedPattern",
    "weave_with_learning",

    # Phase 3: Hot Pattern Feedback
    "HotPatternFeedbackEngine",
    "HotPatternConfig",
    "HotPatternTracker",
    "AdaptiveRetriever",
    "UsageRecord",

    # Phase 4: Advanced Refinement
    "AdvancedRefiner",
    "RefinementStrategy",
    "RefinementResult",
    "QualityMetrics",
    "RefinementPattern",
    "refine_with_strategy",

    # Phase 5: Full Learning Loop
    "FullLearningEngine",
    "ThompsonPriors",
    "PolicyWeights",
    "BackgroundLearner",
    "LearningMetrics",
    "weave_with_full_learning",
]
