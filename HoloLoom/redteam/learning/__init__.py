"""
CARTS Learning System
=====================

Unified adaptive learning for attack strategy selection and payload evolution.

Components:
- LearnerProtocol: Base learning interface
- BanditProtocol: Thompson Sampling interface (backward compatible)
- HeatTrackerProtocol: Heat-based tracking for payloads
- ContextualBanditProtocol: Context-aware strategy selection
- HotPayloadTracker: Heat-based payload tracking

Philosophy: "Learn what works, adapt to what changes."

Author: CARTS Team
Date: 2025-12-03
"""

from .learning_protocols import (
    LearnerProtocol,
    BanditProtocol,
    HeatTrackerProtocol,
    ContextualBanditProtocol,
    ABTestProtocol,
    LearningResult,
    HeatScore,
    ContextKey,
    ABTestResult,
    LearningLevel,
)

from .hot_payloads import (
    PayloadUsageRecord,
    HotPayloadTracker,
    create_hot_payload_tracker,
    DECAY_RATE,
    HOT_THRESHOLD,
    COLD_THRESHOLD,
    BOOST_HOT,
    PENALTY_COLD,
)

from .contextual_bandit import (
    ContextualArm,
    ContextualSelectionResult,
    ContextualRedTeamBandit,
    create_contextual_bandit,
    create_context_key,
)

from .hierarchical_learning import (
    HierarchicalArm,
    HierarchicalSelection,
    HierarchicalUpdate,
    HierarchicalLearner,
    create_hierarchical_learner,
    create_hierarchical_update,
)

from .attack_ab_testing import (
    TestStatus,
    SignificanceMethod,
    TestVariant,
    ABTest,
    ABTestAnalysis,
    AttackABTester,
    create_ab_tester,
    run_quick_ab_test,
)

from .background_learner import (
    LearningEvent,
    BackgroundLearnerConfig,
    BackgroundLearnerStats,
    UpdateableProtocol,
    BackgroundLearner,
    create_background_learner,
)

from .unified_learner import (
    UnifiedSelection,
    UnifiedUpdate,
    UnifiedStats,
    UnifiedLearnerConfig,
    UnifiedLearner,
    create_unified_learner,
    run_learning_demo,
)

__all__ = [
    # Protocols
    'LearnerProtocol',
    'BanditProtocol',
    'HeatTrackerProtocol',
    'ContextualBanditProtocol',
    'ABTestProtocol',

    # Data classes
    'LearningResult',
    'HeatScore',
    'ContextKey',
    'ABTestResult',

    # Enums
    'LearningLevel',

    # Hot Payloads
    'PayloadUsageRecord',
    'HotPayloadTracker',
    'create_hot_payload_tracker',

    # Constants
    'DECAY_RATE',
    'HOT_THRESHOLD',
    'COLD_THRESHOLD',
    'BOOST_HOT',
    'PENALTY_COLD',

    # Contextual Bandit
    'ContextualArm',
    'ContextualSelectionResult',
    'ContextualRedTeamBandit',
    'create_contextual_bandit',
    'create_context_key',

    # Hierarchical Learning
    'HierarchicalArm',
    'HierarchicalSelection',
    'HierarchicalUpdate',
    'HierarchicalLearner',
    'create_hierarchical_learner',
    'create_hierarchical_update',

    # A/B Testing
    'TestStatus',
    'SignificanceMethod',
    'TestVariant',
    'ABTest',
    'ABTestAnalysis',
    'AttackABTester',
    'create_ab_tester',
    'run_quick_ab_test',

    # Background Learner
    'LearningEvent',
    'BackgroundLearnerConfig',
    'BackgroundLearnerStats',
    'UpdateableProtocol',
    'BackgroundLearner',
    'create_background_learner',

    # Unified Learner
    'UnifiedSelection',
    'UnifiedUpdate',
    'UnifiedStats',
    'UnifiedLearnerConfig',
    'UnifiedLearner',
    'create_unified_learner',
    'run_learning_demo',
]

__version__ = '1.5.0'  # Updated for Unified Learner
