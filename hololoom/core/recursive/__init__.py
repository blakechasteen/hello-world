"""
Recursive Learning - Complete 6-Phase Self-Improving System
============================================================

Phases:
1. Scratchpad Integration - Full provenance tracking
2. Loop Engine Integration - Pattern learning from successes
3. Hot Pattern Feedback - Usage-based adaptive retrieval
4. Advanced Refinement - Multi-strategy quality improvement
5. Full Learning Loop - Thompson Sampling + background learning
6. Action Items System - Persistent task tracking with priority learning

Vision: Self-improving knowledge system that learns from usage patterns
and maintains long-term goals across sessions.
"""

from .action_items import (
    ActionCategory,
    ActionItem,
    ActionItemTracker,
    ActionStatus,
    PriorityModel,
    create_action_tracker,
    extract_action_items_from_text,
)
from .advanced_refinement import (
    AdvancedRefiner,
    QualityMetrics,
    RefinementPattern,
    RefinementResult,
    RefinementStrategy,
    refine_with_strategy,
)
from .full_learning_loop import (
    BackgroundLearner,
    FullLearningEngine,
    LearningMetrics,
    PolicyWeights,
    ThompsonPriors,
    weave_with_full_learning,
)
from .hot_patterns import (
    AdaptiveRetriever,
    HotPatternConfig,
    HotPatternFeedbackEngine,
    HotPatternTracker,
    UsageRecord,
)
from .loop_integration import (
    LearnedPattern,
    LearningLoopConfig,
    LearningLoopEngine,
    PatternExtractor,
    PatternLearner,
    weave_with_learning,
)
from .scratchpad_integration import (
    ProvenanceTracker,
    RecursiveRefiner,
    ScratchpadConfig,
    ScratchpadOrchestrator,
    weave_with_scratchpad,
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

    # Phase 6: Action Items System
    "ActionItemTracker",
    "ActionItem",
    "ActionStatus",
    "ActionCategory",
    "PriorityModel",
    "extract_action_items_from_text",
    "create_action_tracker",
]
