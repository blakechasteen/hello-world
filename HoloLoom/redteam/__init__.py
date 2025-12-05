"""
CARTS - Continuous Adversarial Red Team System
==============================================

Self-improving security testing system for HoloLoom's safety framework.

Components:
- AttackStrategy & PayloadGenerator: 12 attack strategies with payload templates
- PayloadMutator: Genetic algorithm for payload evolution
- AttackExecutor: Execute attacks against safety systems
- RedTeamBandit: Thompson Sampling for strategy selection
- VulnerabilityTracker: Track discovered vulnerabilities
- ReportGenerator: Generate Markdown vulnerability reports
- RedTeamOrchestrator: Main orchestrator for continuous testing

Quick Start:
    from HoloLoom.redteam import create_orchestrator

    # Create orchestrator (with or without safety systems)
    orchestrator = create_orchestrator(
        safety_adapter=adapter,  # Optional
        state_dir="./redteam_state"
    )

    # Run a single cycle
    result = await orchestrator.run_cycle(strategies_per_cycle=3)
    print(f"Found {result.vulnerabilities_found} vulnerabilities")

    # Generate report
    report = orchestrator.generate_report()
    print(report)

    # Run continuous testing
    await orchestrator.run_continuous(
        cycle_interval=60.0,
        max_cycles=100
    )

Philosophy:
    "Continuously probe, learn, and evolve."

    CARTS uses Thompson Sampling to learn which attack strategies are most
    effective, genetic algorithms to evolve successful payloads, and
    comprehensive tracking to prevent regressions.

Future Roadmap:
    Phase 2 - Sandbox Isolation (Planned):
        - Sandbox mode for isolated attack execution
        - Process isolation with resource limits
        - Network isolation for external LLM API testing
        - Filesystem sandboxing for state persistence
        - Container-based execution option

    Phase 3 - Sandbox Deployer (Planned):
        - One-click sandbox deployment (Docker/Podman)
        - Kubernetes operator for distributed red teaming
        - CI/CD pipeline integration (GitHub Actions, GitLab CI)
        - Cloud deployment templates (AWS, GCP, Azure)
        - Ephemeral sandbox lifecycle management
        - Cost controls for API-based testing

    Phase 4 - Advanced Features (Future):
        - Multi-agent adversarial swarms
        - Cross-model vulnerability transfer
        - Automated patch generation
        - Real-time defense adaptation
        - Federated learning across deployments

Author: CARTS Team
Date: 2025-12-01
"""

# =============================================================================
# Core Exports
# =============================================================================

# Strategies
from .strategies import (
    AttackStrategy,
    AttackPayload,
    PayloadGenerator,
    create_payload_generator,
)
# Alias for convenience
create_generator = create_payload_generator

# Mutation
from .mutator import (
    MutationType,
    MutationResult,
    CrossoverResult,
    PayloadMutator,
    create_mutator,
)

# Execution
from .executor import (
    AttackOutcome,
    SeverityLevel,
    AttackResult,
    AttackExecutor,
    create_executor,
)

# Bandit
from .bandit import (
    BanditArm,
    SelectionResult,
    RedTeamBandit,
    create_bandit,
)

# Tracking
from .tracker import (
    VulnStatus,
    Vulnerability,
    VulnerabilityTracker,
    create_tracker,
)

# Reporting
from .reporter import (
    ReportSection,
    ReportGenerator,
    generate_report,
    save_report,
)

# Orchestration
from .orchestrator import (
    CycleResult,
    OrchestratorStats,
    RedTeamOrchestrator,
    create_orchestrator,
    run_quick_test,
)

# Learning System (MRF Integration - Phase 2)
from .learning import (
    # Protocols
    LearnerProtocol,
    BanditProtocol,
    HeatTrackerProtocol,
    ContextualBanditProtocol,
    ABTestProtocol,

    # Data classes
    LearningResult,
    HeatScore,
    ContextKey,
    ABTestResult,

    # Enums
    LearningLevel,

    # Hot Payloads
    PayloadUsageRecord,
    HotPayloadTracker,
    create_hot_payload_tracker,

    # Contextual Bandit
    ContextualArm,
    ContextualSelectionResult,
    ContextualRedTeamBandit,
    create_contextual_bandit,
    create_context_key,

    # Hierarchical Learning
    HierarchicalArm,
    HierarchicalSelection,
    HierarchicalUpdate,
    HierarchicalLearner,
    create_hierarchical_learner,
    create_hierarchical_update,

    # A/B Testing
    TestStatus,
    SignificanceMethod,
    TestVariant,
    ABTest,
    ABTestAnalysis,
    AttackABTester,
    create_ab_tester,
    run_quick_ab_test,

    # Background Learner
    LearningEvent,
    BackgroundLearnerConfig,
    BackgroundLearnerStats,
    UpdateableProtocol,
    BackgroundLearner,
    create_background_learner,

    # Unified Learner
    UnifiedSelection,
    UnifiedUpdate,
    UnifiedStats,
    UnifiedLearnerConfig,
    UnifiedLearner,
    create_unified_learner,
    run_learning_demo,
)

# MRF Payload Enhancement (Phase 2)
from .mrf_payloads import (
    # Enums
    EnhancementType,

    # Data Classes
    EnhancementResult,
    MRFPayloadConfig,

    # Main Class
    MRFPayloadEnhancer,

    # Convenience Functions
    create_mrf_enhancer,
    enhance_payload,
    generate_enhanced_payloads,

    # Constants
    STRATEGY_ENHANCEMENTS,
)

# MRF Analytics (Phase 2)
from .mrf_analytics import (
    # Data Classes
    EnhancementMetric,
    StrategyImpact,
    EnhancementTypeImpact,
    ABGroup,
    ABTestResult as MRFABTestResult,  # Avoid conflict with learning.ABTestResult

    # Main Class
    MRFImpactAnalytics,

    # Convenience Functions
    create_analytics,
    log_enhancement_event,
)


# =============================================================================
# All Exports
# =============================================================================

__all__ = [
    # Strategies
    'AttackStrategy',
    'AttackPayload',
    'PayloadGenerator',
    'create_generator',

    # Mutation
    'MutationType',
    'MutationResult',
    'CrossoverResult',
    'PayloadMutator',
    'create_mutator',

    # Execution
    'AttackOutcome',
    'SeverityLevel',
    'AttackResult',
    'AttackExecutor',
    'create_executor',

    # Bandit
    'BanditArm',
    'SelectionResult',
    'RedTeamBandit',
    'create_bandit',

    # Tracking
    'VulnStatus',
    'Vulnerability',
    'VulnerabilityTracker',
    'create_tracker',

    # Reporting
    'ReportSection',
    'ReportGenerator',
    'generate_report',
    'save_report',

    # Orchestration
    'CycleResult',
    'OrchestratorStats',
    'RedTeamOrchestrator',
    'create_orchestrator',
    'run_quick_test',

    # Learning System (MRF Integration)
    'LearnerProtocol',
    'BanditProtocol',
    'HeatTrackerProtocol',
    'ContextualBanditProtocol',
    'ABTestProtocol',
    'LearningResult',
    'HeatScore',
    'ContextKey',
    'ABTestResult',
    'LearningLevel',
    'PayloadUsageRecord',
    'HotPayloadTracker',
    'create_hot_payload_tracker',

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


# =============================================================================
# Version Info
# =============================================================================

__version__ = '1.6.0'  # Updated for Unified Learner
__author__ = 'CARTS Team'
