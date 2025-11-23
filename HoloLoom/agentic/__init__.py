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
    from HoloLoom.agentic import create_agentic_orchestrator, ReasoningMode

    async with create_agentic_orchestrator(config, shards) as agent:
        result = await agent.reason(
            query=Query(text="Is Thompson Sampling optimal here?"),
            mode=ReasoningMode.VERIFY
        )

        print(f"Verified: {result.verification.verified}")
        print(f"Steps: {result.total_queries}")
"""

from .core import (
    AgenticOrchestrator,
    AgenticIntent,
    AgenticResult,
    ReasoningMode,
    VerificationResult,
    create_agentic_orchestrator,
)

from .web_research import (
    WebResearchOrchestrator,
    WebResearchResult,
)

from .embedding_integrity import (
    EmbeddingRun,
    VectorMeta,
    EmbeddingIntegrityMonitor,
    DeterminismCheck,
    QualityMetrics,
)

from .skills import (
    execute_skill,
    list_available_skills,
    get_registry,
    SkillRegistry,
    SkillExecutor,
    SkillTemplate,
    SkillExecutionResult,
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
]