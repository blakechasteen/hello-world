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
    from hololoom.redteam import create_orchestrator

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

# Strategies (from strategies.py module)
from . import strategies as _strategies_module

AttackStrategy = _strategies_module.AttackStrategy
AttackPayload = _strategies_module.AttackPayload
PayloadGenerator = _strategies_module.PayloadGenerator
create_payload_generator = _strategies_module.create_payload_generator

# Alias for convenience
create_generator = create_payload_generator

# Strategy Generators (from strategy_generators/ package)
# =============================================================================
# NEW - Phase 2+: Sandbox, Swarm, Refinement, Probes (November 2025)
# =============================================================================
# Phase 2: Sandbox Isolation
from . import sandbox as _sandbox_module
from . import strategy_generators

# Bandit
from .bandit import (
    BanditArm,
    RedTeamBandit,
    SelectionResult,
    create_bandit,
)

# Execution
from .executor import (
    AttackExecutor,
    AttackOutcome,
    AttackResult,
    SeverityLevel,
    create_executor,
)

# Learning System (MRF Integration - Phase 2)
from .learning import (
    ABTest,
    ABTestAnalysis,
    ABTestProtocol,
    ABTestResult,
    AttackABTester,
    BackgroundLearner,
    BackgroundLearnerConfig,
    BackgroundLearnerStats,
    BanditProtocol,
    ContextKey,
    # Contextual Bandit
    ContextualArm,
    ContextualBanditProtocol,
    ContextualRedTeamBandit,
    ContextualSelectionResult,
    HeatScore,
    HeatTrackerProtocol,
    # Hierarchical Learning
    HierarchicalArm,
    HierarchicalLearner,
    HierarchicalSelection,
    HierarchicalUpdate,
    HotPayloadTracker,
    # Protocols
    LearnerProtocol,
    # Background Learner
    LearningEvent,
    # Enums
    LearningLevel,
    # Data classes
    LearningResult,
    # Hot Payloads
    PayloadUsageRecord,
    SignificanceMethod,
    # A/B Testing
    TestStatus,
    TestVariant,
    UnifiedLearner,
    UnifiedLearnerConfig,
    # Unified Learner
    UnifiedSelection,
    UnifiedStats,
    UnifiedUpdate,
    UpdateableProtocol,
    create_ab_tester,
    create_background_learner,
    create_context_key,
    create_contextual_bandit,
    create_hierarchical_learner,
    create_hierarchical_update,
    create_hot_payload_tracker,
    create_unified_learner,
    run_learning_demo,
    run_quick_ab_test,
)

# MRF Analytics (Phase 2)
from .mrf_analytics import (
    ABGroup,
    # Data Classes
    EnhancementMetric,
    EnhancementTypeImpact,
    # Main Class
    MRFImpactAnalytics,
    StrategyImpact,
    # Convenience Functions
    create_analytics,
    log_enhancement_event,
)
from .mrf_analytics import (
    ABTestResult as MRFABTestResult,  # Avoid conflict with learning.ABTestResult
)

# MRF Payload Enhancement (Phase 2)
from .mrf_payloads import (
    # Constants
    STRATEGY_ENHANCEMENTS,
    # Data Classes
    EnhancementResult,
    # Enums
    EnhancementType,
    MRFPayloadConfig,
    # Main Class
    MRFPayloadEnhancer,
    # Convenience Functions
    create_mrf_enhancer,
    enhance_payload,
    generate_enhanced_payloads,
)

# Mutation
from .mutator import (
    CrossoverResult,
    MutationResult,
    MutationType,
    PayloadMutator,
    create_mutator,
)

# Orchestration
from .orchestrator import (
    CycleResult,
    OrchestratorStats,
    RedTeamOrchestrator,
    create_orchestrator,
    run_quick_test,
)

# Reporting
from .reporter import (
    ReportGenerator,
    ReportSection,
    generate_report,
    save_report,
)

# Tracking
from .tracker import (
    Vulnerability,
    VulnerabilityTracker,
    VulnStatus,
    create_tracker,
)

SandboxMode = _sandbox_module.SandboxMode
SandboxConfig = _sandbox_module.SandboxConfig
SandboxResult = _sandbox_module.SandboxResult
SandboxedExecutor = _sandbox_module.SandboxedExecutor
create_sandboxed_executor = _sandbox_module.create_sandboxed_executor

# Phase 3: Swarm Coordination
from . import swarm as _swarm_module

SwarmCoordinator = _swarm_module.SwarmCoordinator
MessageBus = _swarm_module.MessageBus
BaseAgent = _swarm_module.BaseAgent
ScoutAgent = _swarm_module.ScoutAgent
AttackerAgent = _swarm_module.AttackerAgent
ExploiterAgent = _swarm_module.ExploiterAgent
CoordinatorAgent = _swarm_module.CoordinatorAgent
create_scout_agent = _swarm_module.create_scout_agent
create_attacker_agent = _swarm_module.create_attacker_agent
create_exploiter_agent = _swarm_module.create_exploiter_agent
create_coordinator_agent = _swarm_module.create_coordinator_agent

# Phase 4: Attack Refinement
from . import refinement as _refinement_module

AttackRefiner = _refinement_module.AttackRefiner
QualityTrajectoryTracker = _refinement_module.QualityTrajectoryTracker
AttackRefinementStrategy = _refinement_module.AttackRefinementStrategy
AttackRefinementResult = _refinement_module.AttackRefinementResult

# Phase 5: Behavioral Probes
from . import probes as _probes_module

AttackProber = _probes_module.AttackProber
AttackProbe = _probes_module.AttackProbe
ProbeResult = _probes_module.ProbeResult
VulnerabilityProbeReport = _probes_module.VulnerabilityProbeReport


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

    # =================================================================
    # Phase 2+: Sandbox, Swarm, Refinement, Probes (NEW - Nov 2025)
    # =================================================================

    # Phase 2: Sandbox Isolation
    'SandboxMode',
    'SandboxConfig',
    'SandboxResult',
    'SandboxedExecutor',
    'create_sandboxed_executor',

    # Phase 3: Swarm Coordination
    'SwarmCoordinator',
    'MessageBus',
    'BaseAgent',
    'ScoutAgent',
    'AttackerAgent',
    'ExploiterAgent',
    'CoordinatorAgent',
    'create_scout_agent',
    'create_attacker_agent',
    'create_exploiter_agent',
    'create_coordinator_agent',

    # Phase 4: Attack Refinement
    'AttackRefiner',
    'QualityTrajectoryTracker',
    'AttackRefinementStrategy',
    'AttackRefinementResult',

    # Phase 5: Behavioral Probes
    'AttackProber',
    'AttackProbe',
    'ProbeResult',
    'VulnerabilityProbeReport',
]


# =============================================================================
# Version Info
# =============================================================================

__version__ = '1.6.0'  # Updated for Unified Learner
__author__ = 'CARTS Team'
