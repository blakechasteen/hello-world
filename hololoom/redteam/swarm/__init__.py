"""Multi-agent swarm for coordinated red team attacks.

CARTS Phase 4: Multi-Agent Swarm System
Status: Foundation Implementation (December 2025)
Location: HoloLoom/redteam/swarm/
Performance Target: <10ms message latency

FIRST PRINCIPLE: "Safety is not a constraint on effectiveness;
                  it is a prerequisite for it."

This module provides a safety-first architecture for coordinated red team
operations with specialized agent roles:
- Scout: Surface probing and vulnerability discovery
- Attacker: Attack execution with Thompson Sampling strategy selection
- Exploiter: Vulnerability exploitation and privilege escalation
- Coordinator: Swarm coordination and task distribution

Safety Architecture (5-layer defense in depth):
- Layer 1: Authorization (NO BYPASS. NO EXCEPTIONS.)
- Layer 2: Scope validation (whitelist-only targets)
- Layer 3: Rate limiting (UPPER BOUNDS, not suggestions)
- Layer 4: Audit logging (immutable, cannot be disabled)
- Layer 5: Anomaly detection (behavioral pattern tracking)

Communication Architecture:
- Protocol-based design (AgentProtocol, CoordinatorProtocol)
- Async-first message bus with priority queuing
- Dead letter queue for failed messages
- Per-agent message tracking and acknowledgments
- Metrics collection for performance monitoring
"""

from .ab_testing import (
    # A/B Test Manager
    ABTestManager,
    # Early Stopping
    EarlyStoppingChecker,
    EffectSizeCategory,
    Experiment,
    ExperimentConfig,
    # Enums
    ExperimentStatus,
    SignificanceLevel,
    # Statistical Analysis
    StatisticalAnalyzer,
    StatisticalResult,
    TrafficAllocation,
    # Traffic Splitting
    TrafficSplitter,
    # Data classes
    Variant,
    # Factory functions
    create_ab_test_manager,
    create_experiment_config,
    run_ab_test,
)
from .agents import (
    AttackerAgent,
    AttackOutcome,
    AttackStrategy,
    # Attacker agent (attack execution with Thompson Sampling)
    AttackStrategyType,
    # Base agent
    BaseAgent,
    # Coordinator agent (task distribution, result aggregation)
    CampaignStatus,
    CoordinatorAgent,
    Discovery,
    # Scout agent (surface probing, vulnerability discovery)
    DiscoveryType,
    ExploitationTechnique,
    # Exploiter agent (vulnerability exploitation)
    ExploitationType,
    ExploiterAgent,
    ExploitResult,
    ScoutAgent,
    TaskAssignment,
    create_attacker_agent,
    create_coordinator_agent,
    create_exploiter_agent,
    create_scout_agent,
)
from .communication import MessageBus
from .coordinator import (
    CampaignPhase,
    SwarmCampaignResult,
    SwarmCoordinator,
    SwarmMetrics,
)
from .learning import (
    BackgroundLearner,
    # Timescale 3: Per-Cycle (~minutes)
    CrossStrategyInsight,
    # Unified Coordinator
    HierarchicalLearningCoordinator,
    LearnedPattern,
    LearningEvent,
    # Timescales
    LearningTimescale,
    PatternUpdate,
    # Timescale 1: Per-Attack (immediate)
    PayloadHeat,
    PerAttackLearner,
    PerCycleLearner,
    PerTaskLearner,
    # Timescale 4: Background (~hours)
    SystemPrior,
    # Timescale 2: Per-Task (~seconds)
    ThompsonSamplingPrior,
    create_learning_coordinator,
)
from .protocols import (
    AgentMessage,
    AgentProtocol,
    AgentResult,
    AgentRole,
    AgentState,
    AgentTask,
    CoordinatorProtocol,
    MessagePriority,
)
from .safety import (
    # Layer 5: Anomaly detection
    AnomalyDetector,
    AuditEntry,
    AuditEventType,
    # Layer 4: Audit logging
    AuditLogger,
    # Exceptions
    AuthorizationError,
    # Layer 1: Authorization
    AuthorizationManager,
    # Data classes
    AuthorizationToken,
    RateLimitConfig,
    # Layer 3: Rate limiting
    RateLimiter,
    RateLimitExceededError,
    RateLimitState,
    # Integrated gate
    SafetyGate,
    # Layer 2: Scope validation
    ScopeValidator,
    ScopeViolationError,
    # Severity assessment
    SeverityAssessor,
    # Enums
    SeverityLevel,
    create_authorization_token,
    # Factory functions
    create_safety_gate,
)

__all__ = [
    # Swarm Coordinator (main orchestrator)
    "SwarmCoordinator",
    "SwarmMetrics",
    "SwarmCampaignResult",
    # Protocol Enums
    "AgentRole",
    "AgentState",
    "MessagePriority",
    # Protocol Data classes
    "AgentMessage",
    "AgentTask",
    "AgentResult",
    # Protocols
    "AgentProtocol",
    "CoordinatorProtocol",
    # Communication
    "MessageBus",
    # Safety Enums
    "SeverityLevel",
    "AuditEventType",
    # Safety Data classes
    "AuthorizationToken",
    "AuditEntry",
    "RateLimitConfig",
    "RateLimitState",
    # Safety Exceptions
    "AuthorizationError",
    "ScopeViolationError",
    "RateLimitExceededError",
    # Safety Layers
    "AuthorizationManager",
    "ScopeValidator",
    "RateLimiter",
    "AuditLogger",
    "AnomalyDetector",
    # Safety Assessment
    "SeverityAssessor",
    # Integrated Safety Gate
    "SafetyGate",
    # Safety Factory functions
    "create_safety_gate",
    "create_authorization_token",
    # Base Agent
    "BaseAgent",
    # Scout Agent (surface probing, vulnerability discovery)
    "DiscoveryType",
    "Discovery",
    "ScoutAgent",
    "create_scout_agent",
    # Attacker Agent (attack execution with Thompson Sampling)
    "AttackStrategyType",
    "AttackStrategy",
    "AttackOutcome",
    "AttackerAgent",
    "create_attacker_agent",
    # Exploiter Agent (vulnerability exploitation)
    "ExploitationType",
    "ExploitResult",
    "ExploitationTechnique",
    "ExploiterAgent",
    "create_exploiter_agent",
    # Coordinator Agent (task distribution, result aggregation)
    "CampaignPhase",
    "CampaignStatus",
    "TaskAssignment",
    "CoordinatorAgent",
    "create_coordinator_agent",
    # Learning Timescales
    "LearningTimescale",
    "LearningEvent",
    "PatternUpdate",
    # Timescale 1: Per-Attack (immediate)
    "PayloadHeat",
    "PerAttackLearner",
    # Timescale 2: Per-Task (~seconds)
    "ThompsonSamplingPrior",
    "PerTaskLearner",
    # Timescale 3: Per-Cycle (~minutes)
    "CrossStrategyInsight",
    "PerCycleLearner",
    # Timescale 4: Background (~hours)
    "SystemPrior",
    "LearnedPattern",
    "BackgroundLearner",
    # Unified Learning Coordinator
    "HierarchicalLearningCoordinator",
    "create_learning_coordinator",
    # A/B Testing Enums
    "ExperimentStatus",
    "TrafficAllocation",
    "SignificanceLevel",
    "EffectSizeCategory",
    # A/B Testing Data Classes
    "Variant",
    "StatisticalResult",
    "ExperimentConfig",
    "Experiment",
    # A/B Testing Components
    "StatisticalAnalyzer",
    "TrafficSplitter",
    "EarlyStoppingChecker",
    "ABTestManager",
    # A/B Testing Factory Functions
    "create_ab_test_manager",
    "create_experiment_config",
    "run_ab_test",
]

__version__ = "1.4.0"  # Bumped for A/B testing framework
__status__ = "Production Ready"
