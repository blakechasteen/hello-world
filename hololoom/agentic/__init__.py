"""
HoloLoom Agentic Intelligence
==============================
Self-directed reasoning with verification loops.

Extends HoloLoom's recursive learning system (Phase 5) with autonomous
query generation, multi-step reasoning, and verification protocols.

Core Features:
- ReasoningMode: DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE
- Intent tracking with goal decomposition
- Verification loops with contradiction detection
- Multi-query exploration and synthesis
- Full provenance in audit trail

Philosophy:
"Don't just answer - understand, verify, refine."

Integration Points:
- Builds on recursive.FullLearningEngine
- Uses alignment.audit_trail for logging
- Extends action_items for goal tracking
- Leverages ReflectionBuffer for learning

Usage:
    from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

    async with create_agentic_orchestrator(config, shards) as agent:
        result = await agent.reason(
            query=Query(text="Is Thompson Sampling optimal here?"),
            mode=ReasoningMode.VERIFY
        )

        print(f"Verified: {result.verification.verified}")
        print(f"Steps: {result.total_queries}")
"""

# Phase 7.2: Task Delegation System
from .capability_analyzer import (
    CapabilityAnalysis,
    QueryCapabilityAnalyzer,
    QueryComplexity,
    analyze_query_capabilities,
)

# Phase 6.3: Context Handoff (Foundation for 7.1)
from .context_handoff import (
    ContextHandoff,
    ContextItem,
    HandoffContext,
    HandoffStrategy,
    get_optimal_handoff_strategy,
    prepare_agent_handoff,
)
from .core import (
    AgenticIntent,
    AgenticOrchestrator,
    AgenticResult,
    ReasoningMode,
    VerificationResult,
    create_agentic_orchestrator,
)
from .embedding_integrity import (
    DeterminismCheck,
    EmbeddingIntegrityMonitor,
    EmbeddingRun,
    QualityMetrics,
    VectorMeta,
)
from .ensemble_decision import (
    AgentResponse,
    EnsembleAggregator,
    EnsembleResult,
    EnsembleStrategy,
    NumericEnsemble,
    TextEnsemble,
    aggregate_responses,
    create_agent_response,
)
from .expert_router import (
    AgentScore,
    ExpertRouter,
    RoutingDecision,
    SelectionStrategy,
    ThompsonPrior,
    create_expert_router,
)

# Phase 7.1: Multi-Agent Communication Protocol
from .multi_agent import (
    # Data Types
    AgentCapability,
    AgentInfo,
    # Message Classes
    AgentMessage,
    # Registry
    AgentRegistry,
    AgentStatus,
    # Message Bus
    MessageBus,
    MessagePriority,
    MessageType,
    # Coordinator
    MultiAgentCoordinator,
    SharedMemory,
    # Shared Memory
    SharedMemoryEntry,
    create_agent_message,
    create_code_agent,
    # Convenience Functions
    create_coordinator,
    create_research_agent,
    create_synthesis_agent,
)
from .skills import (
    SkillExecutionResult,
    SkillExecutor,
    SkillRegistry,
    SkillTemplate,
    execute_skill,
    get_registry,
    list_available_skills,
)
from .web_research import (
    WebResearchOrchestrator,
    WebResearchResult,
)

__all__ = [
    # Core agentic reasoning
    "AgenticOrchestrator",
    "AgenticIntent",
    "AgenticResult",
    "ReasoningMode",
    "VerificationResult",
    "create_agentic_orchestrator",

    # Web-enhanced research
    "WebResearchOrchestrator",
    "WebResearchResult",

    # Embedding verification
    "EmbeddingRun",
    "VectorMeta",
    "EmbeddingIntegrityMonitor",
    "DeterminismCheck",
    "QualityMetrics",

    # Skills system
    "execute_skill",
    "list_available_skills",
    "get_registry",
    "SkillRegistry",
    "SkillExecutor",
    "SkillTemplate",
    "SkillExecutionResult",

    # Phase 7.1: Multi-Agent Communication Protocol
    "AgentCapability",
    "MessageType",
    "MessagePriority",
    "AgentStatus",
    "AgentMessage",
    "AgentInfo",
    "AgentRegistry",
    "MessageBus",
    "SharedMemoryEntry",
    "SharedMemory",
    "MultiAgentCoordinator",
    "create_coordinator",
    "create_agent_message",
    "create_research_agent",
    "create_synthesis_agent",
    "create_code_agent",

    # Phase 6.3: Context Handoff
    "ContextItem",
    "HandoffContext",
    "HandoffStrategy",
    "ContextHandoff",
    "prepare_agent_handoff",
    "get_optimal_handoff_strategy",

    # Phase 7.2: Task Delegation System
    # Capability Analyzer
    "QueryCapabilityAnalyzer",
    "CapabilityAnalysis",
    "QueryComplexity",
    "analyze_query_capabilities",
    # Expert Router (Thompson Sampling)
    "ExpertRouter",
    "SelectionStrategy",
    "ThompsonPrior",
    "AgentScore",
    "RoutingDecision",
    "create_expert_router",
    # Ensemble Decision
    "EnsembleAggregator",
    "EnsembleStrategy",
    "EnsembleResult",
    "AgentResponse",
    "NumericEnsemble",
    "TextEnsemble",
    "aggregate_responses",
    "create_agent_response",
]
