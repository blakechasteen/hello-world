"""
MRF Analytics Package
====================
Production analytics system for Metaprompting Refinement Framework.

**Phase 2 - November 2025**: Complete analytics suite with dashboard,
Thompson Sampling learning, and A/B testing.

Components:
- MRFDashboard: Real-time performance monitoring
- ThompsonLearner: Continuous learning for strategy selection
- ABTest: Statistical validation framework

Usage:
    from hololoom.prompting.analytics import (
        MRFDashboard,
        ThompsonLearner,
        ABTest,
        create_dashboard
    )

    # Create integrated dashboard
    dashboard = create_dashboard(
        enable_learning=True,
        enable_ab_testing=True
    )

    # Or create components separately
    dashboard = MRFDashboard()
    learner = ThompsonLearner()
    ab_test = ABTest(
        name="test_name",
        control_description="Baseline",
        treatment_description="Enhanced"
    )
"""

# Dashboard
from hololoom.prompting.analytics.dashboard import (
    MRFDashboard,
    MRFEnhancementLog,
    SystemStatistics,
    create_dashboard,
)

# Thompson Sampling Learning (optional)
try:
    from hololoom.prompting.analytics.learning import (
        QueryTypeProfile,
        StrategyStats,
        ThompsonLearner,
        create_learner,
    )
    LEARNING_AVAILABLE = True
except ImportError:
    ThompsonLearner = None
    StrategyStats = None
    QueryTypeProfile = None
    create_learner = None
    LEARNING_AVAILABLE = False

# A/B Testing (optional)
try:
    from hololoom.prompting.analytics.ab_testing import (
        ABGroup,
        ABResult,
        ABTest,
        ABTestConfig,
        DeploymentDecision,
        create_ab_test,
    )
    AB_TESTING_AVAILABLE = True
except ImportError:
    ABTest = None
    ABGroup = None
    ABResult = None
    ABTestConfig = None
    DeploymentDecision = None
    create_ab_test = None
    AB_TESTING_AVAILABLE = False

__all__ = [
    # Dashboard
    "MRFDashboard",
    "MRFEnhancementLog",
    "SystemStatistics",
    "create_dashboard",

    # Learning
    "ThompsonLearner",
    "StrategyStats",
    "QueryTypeProfile",
    "create_learner",
    "LEARNING_AVAILABLE",

    # A/B Testing
    "ABTest",
    "ABGroup",
    "ABResult",
    "ABTestConfig",
    "DeploymentDecision",
    "create_ab_test",
    "AB_TESTING_AVAILABLE",
]
