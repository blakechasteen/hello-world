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

from .attack_ab_testing import (
    ABTest,
    ABTestAnalysis,
    AttackABTester,
    SignificanceMethod,
    TestStatus,
    TestVariant,
    create_ab_tester,
    run_quick_ab_test,
)
from .background_learner import (
    BackgroundLearner,
    BackgroundLearnerConfig,
    BackgroundLearnerStats,
    LearningEvent,
    UpdateableProtocol,
    create_background_learner,
)
from .contextual_bandit import (
    ContextualArm,
    ContextualRedTeamBandit,
    ContextualSelectionResult,
    create_context_key,
    create_contextual_bandit,
)
from .hierarchical_learning import (
    HierarchicalArm,
    HierarchicalLearner,
    HierarchicalSelection,
    HierarchicalUpdate,
    create_hierarchical_learner,
    create_hierarchical_update,
)
from .hot_payloads import (
    BOOST_HOT,
    COLD_THRESHOLD,
    DECAY_RATE,
    HOT_THRESHOLD,
    PENALTY_COLD,
    HotPayloadTracker,
    PayloadUsageRecord,
    create_hot_payload_tracker,
)
from .learning_protocols import (
    ABTestProtocol,
    ABTestResult,
    BanditProtocol,
    ContextKey,
    ContextualBanditProtocol,
    HeatScore,
    HeatTrackerProtocol,
    LearnerProtocol,
    LearningLevel,
    LearningResult,
)
from .unified_learner import (
    UnifiedLearner,
    UnifiedLearnerConfig,
    UnifiedSelection,
    UnifiedStats,
    UnifiedUpdate,
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
