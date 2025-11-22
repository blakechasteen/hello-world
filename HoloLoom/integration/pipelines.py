"""
Predefined Pipeline Templates for HoloLoom Integration Framework

Provides standard pipeline definitions for common use cases:
- SIMPLE: Fast query answering (~100ms)
- VERIFIED: Answer with verification (~500ms)
- QUALITY_ASSURED: Answer with QA checks (~700ms)
- COLLABORATIVE: Multi-agent reasoning (~750ms)
- COMPREHENSIVE: All systems active (~1000ms)

Author: HoloLoom Team
Date: 2025-01-21
"""

from HoloLoom.integration.framework import (
    PipelineDefinition,
    StageDefinition,
    StageType
)


# ============================================================================
# Pipeline 1: Simple Query (BARE Mode)
# ============================================================================

PIPELINE_SIMPLE = PipelineDefinition(
    name="simple_query",
    default_mode="BARE",
    enable_parallel=False,  # Sequential for speed
    max_total_timeout_ms=5000,  # 5 seconds max
    stages=[
        StageDefinition(
            stage_type=StageType.RETRIEVAL,
            department_name="context",
            timeout_ms=2000,
            required=True,
            confidence_threshold=0.5,
            metadata={"max_memories": 5, "fast_path": True}
        ),
        StageDefinition(
            stage_type=StageType.REASONING,
            department_name="orchestrator",
            timeout_ms=2000,
            required=True,
            confidence_threshold=0.5,
            metadata={"mode": "BARE"}
        ),
        StageDefinition(
            stage_type=StageType.OUTPUT,
            department_name="orchestrator",
            timeout_ms=1000,
            required=True,
            metadata={"format": "text"}
        ),
    ],
    metadata={
        "description": "Fast query answering with minimal processing",
        "target_latency_ms": 100,
        "use_case": "Simple factual queries, quick lookups"
    }
)


# ============================================================================
# Pipeline 2: Verified Answer (FAST Mode)
# ============================================================================

PIPELINE_VERIFIED = PipelineDefinition(
    name="verified_answer",
    default_mode="FAST",
    enable_parallel=True,
    max_total_timeout_ms=15000,  # 15 seconds max
    stages=[
        StageDefinition(
            stage_type=StageType.RETRIEVAL,
            department_name="context",
            timeout_ms=3000,
            required=True,
            confidence_threshold=0.6,
            metadata={"max_memories": 10}
        ),
        StageDefinition(
            stage_type=StageType.REASONING,
            department_name="orchestrator",
            timeout_ms=5000,
            required=True,
            confidence_threshold=0.6,
            metadata={"mode": "FAST"}
        ),
        StageDefinition(
            stage_type=StageType.VERIFICATION,
            department_name="verification",
            timeout_ms=5000,
            required=False,  # Optional - adds quality but not critical
            confidence_threshold=0.7,
            metadata={"check_contradictions": True, "check_completeness": True}
        ),
        StageDefinition(
            stage_type=StageType.OUTPUT,
            department_name="orchestrator",
            timeout_ms=2000,
            required=True,
            metadata={"format": "text", "include_sources": True}
        ),
    ],
    metadata={
        "description": "Answer with verification for accuracy",
        "target_latency_ms": 500,
        "use_case": "Claims needing verification, important queries"
    }
)


# ============================================================================
# Pipeline 3: Quality Assured (FUSED Mode with QA)
# ============================================================================

PIPELINE_QUALITY_ASSURED = PipelineDefinition(
    name="quality_assured",
    default_mode="FUSED",
    enable_parallel=True,
    max_total_timeout_ms=30000,  # 30 seconds max
    stages=[
        StageDefinition(
            stage_type=StageType.RETRIEVAL,
            department_name="context",
            timeout_ms=5000,
            required=True,
            confidence_threshold=0.6,
            metadata={"max_memories": 20}
        ),
        StageDefinition(
            stage_type=StageType.REASONING,
            department_name="orchestrator",
            timeout_ms=10000,
            required=True,
            confidence_threshold=0.7,
            metadata={"mode": "FUSED"}
        ),
        # QA and Verification run in parallel
        StageDefinition(
            stage_type=StageType.QUALITY_ASSURANCE,
            department_name="quality_assurance",
            timeout_ms=10000,
            required=False,
            parallel_with=["verification"],  # Can run with verification
            metadata={
                "check_slop": True,
                "check_logic": True,
                "auto_fix": False  # Just detect, don't fix yet
            }
        ),
        StageDefinition(
            stage_type=StageType.VERIFICATION,
            department_name="verification",
            timeout_ms=5000,
            required=False,
            parallel_with=["quality_assurance"],  # Can run with QA
            metadata={"check_contradictions": True, "check_completeness": True}
        ),
        # Guidance runs after QA/verification
        StageDefinition(
            stage_type=StageType.GUIDANCE,
            department_name="guidance",
            timeout_ms=5000,
            required=False,
            metadata={"suggest_improvements": True}
        ),
        StageDefinition(
            stage_type=StageType.OUTPUT,
            department_name="orchestrator",
            timeout_ms=2000,
            required=True,
            metadata={
                "format": "text",
                "include_sources": True,
                "include_qa_report": True
            }
        ),
    ],
    metadata={
        "description": "Answer with quality assurance and verification",
        "target_latency_ms": 700,
        "use_case": "Code generation, critical responses, production systems"
    }
)


# ============================================================================
# Pipeline 4: Collaborative Multi-Agent (FUSED Mode)
# ============================================================================

PIPELINE_COLLABORATIVE = PipelineDefinition(
    name="collaborative_multi_agent",
    default_mode="FUSED",
    enable_parallel=True,
    max_total_timeout_ms=30000,  # 30 seconds max
    stages=[
        StageDefinition(
            stage_type=StageType.RETRIEVAL,
            department_name="context",
            timeout_ms=5000,
            required=True,
            confidence_threshold=0.6,
            metadata={"max_memories": 20}
        ),
        # Multiple reasoning engines in parallel
        StageDefinition(
            stage_type=StageType.REASONING,
            department_name="orchestrator",
            timeout_ms=10000,
            required=True,
            parallel_with=["collaborative_agents"],  # Can run with collab agents
            metadata={"mode": "FUSED"}
        ),
        StageDefinition(
            stage_type=StageType.REASONING,
            department_name="collaborative_agents",
            timeout_ms=10000,
            required=False,
            parallel_with=["orchestrator"],  # Can run with orchestrator
            metadata={
                "num_agents": 3,
                "consensus_threshold": 0.7,
                "enable_debate": True
            }
        ),
        # Consensus building and verification
        StageDefinition(
            stage_type=StageType.COORDINATION,
            department_name="collaborative_agents",
            timeout_ms=5000,
            required=False,
            metadata={"build_consensus": True}
        ),
        StageDefinition(
            stage_type=StageType.VERIFICATION,
            department_name="verification",
            timeout_ms=5000,
            required=False,
            metadata={"check_consensus": True}
        ),
        StageDefinition(
            stage_type=StageType.OUTPUT,
            department_name="orchestrator",
            timeout_ms=2000,
            required=True,
            metadata={
                "format": "text",
                "include_sources": True,
                "include_consensus_report": True
            }
        ),
    ],
    metadata={
        "description": "Multi-agent collaborative reasoning with consensus",
        "target_latency_ms": 750,
        "use_case": "Complex problems, research queries, multi-perspective analysis"
    }
)


# ============================================================================
# Pipeline 5: Comprehensive (All Systems Active)
# ============================================================================

PIPELINE_COMPREHENSIVE = PipelineDefinition(
    name="comprehensive",
    default_mode="FUSED",
    enable_parallel=True,
    enable_graceful_degradation=True,
    max_total_timeout_ms=60000,  # 60 seconds max
    stages=[
        # Phase 1: Context gathering
        StageDefinition(
            stage_type=StageType.RETRIEVAL,
            department_name="context",
            timeout_ms=5000,
            required=True,
            confidence_threshold=0.6,
            metadata={"max_memories": 30, "enable_shuttle": True}
        ),
        # Phase 2: Planning (optional)
        StageDefinition(
            stage_type=StageType.PLANNING,
            department_name="planning",
            timeout_ms=5000,
            required=False,
            metadata={"decompose_goals": True}
        ),
        # Phase 3: Multi-path reasoning (parallel)
        StageDefinition(
            stage_type=StageType.REASONING,
            department_name="orchestrator",
            timeout_ms=15000,
            required=True,
            parallel_with=["collaborative_agents", "reasoning"],
            metadata={"mode": "FUSED"}
        ),
        StageDefinition(
            stage_type=StageType.REASONING,
            department_name="collaborative_agents",
            timeout_ms=15000,
            required=False,
            parallel_with=["orchestrator", "reasoning"],
            metadata={"num_agents": 5, "enable_debate": True}
        ),
        StageDefinition(
            stage_type=StageType.REASONING,
            department_name="reasoning",
            timeout_ms=15000,
            required=False,
            parallel_with=["orchestrator", "collaborative_agents"],
            metadata={"strategies": ["deductive", "abductive", "analogical"]}
        ),
        # Phase 4: Quality checks (parallel)
        StageDefinition(
            stage_type=StageType.QUALITY_ASSURANCE,
            department_name="quality_assurance",
            timeout_ms=10000,
            required=False,
            parallel_with=["verification"],
            metadata={"comprehensive_checks": True, "auto_fix": True}
        ),
        StageDefinition(
            stage_type=StageType.VERIFICATION,
            department_name="verification",
            timeout_ms=10000,
            required=False,
            parallel_with=["quality_assurance"],
            metadata={"comprehensive_checks": True}
        ),
        # Phase 5: Guidance and learning
        StageDefinition(
            stage_type=StageType.GUIDANCE,
            department_name="guidance",
            timeout_ms=5000,
            required=False,
            metadata={"comprehensive_guidance": True}
        ),
        StageDefinition(
            stage_type=StageType.LEARNING,
            department_name="recursive_learning",
            timeout_ms=5000,
            required=False,
            metadata={"extract_patterns": True, "update_priors": True}
        ),
        # Phase 6: Output
        StageDefinition(
            stage_type=StageType.OUTPUT,
            department_name="orchestrator",
            timeout_ms=2000,
            required=True,
            metadata={
                "format": "text",
                "include_sources": True,
                "include_full_report": True
            }
        ),
    ],
    metadata={
        "description": "All systems active for maximum quality and insight",
        "target_latency_ms": 1000,
        "use_case": "Research, critical decisions, comprehensive analysis"
    }
)


# ============================================================================
# Pipeline 6: Shuttle-Optimized (MCTS Memory Retrieval)
# ============================================================================

PIPELINE_SHUTTLE_OPTIMIZED = PipelineDefinition(
    name="shuttle_optimized",
    default_mode="FUSED",
    enable_parallel=True,
    max_total_timeout_ms=20000,  # 20 seconds max
    stages=[
        # Shuttle-based retrieval (Warp + Yarn with MCTS)
        StageDefinition(
            stage_type=StageType.RETRIEVAL,
            department_name="shuttle_retrieval",
            timeout_ms=8000,
            required=True,
            confidence_threshold=0.7,
            metadata={
                "enable_mcts": True,
                "max_depth": 3,
                "thompson_sampling": True
            }
        ),
        StageDefinition(
            stage_type=StageType.REASONING,
            department_name="orchestrator",
            timeout_ms=8000,
            required=True,
            metadata={"mode": "FUSED"}
        ),
        StageDefinition(
            stage_type=StageType.VERIFICATION,
            department_name="verification",
            timeout_ms=5000,
            required=False,
            metadata={"check_graph_consistency": True}
        ),
        StageDefinition(
            stage_type=StageType.OUTPUT,
            department_name="orchestrator",
            timeout_ms=2000,
            required=True,
            metadata={"format": "text", "include_graph_trace": True}
        ),
    ],
    metadata={
        "description": "MCTS-optimized memory retrieval with Shuttle system",
        "target_latency_ms": 600,
        "use_case": "Complex knowledge graph queries, connected information discovery"
    }
)


# ============================================================================
# Pipeline Registry
# ============================================================================

# All available pipelines
PIPELINES = {
    "simple": PIPELINE_SIMPLE,
    "verified": PIPELINE_VERIFIED,
    "quality_assured": PIPELINE_QUALITY_ASSURED,
    "collaborative": PIPELINE_COLLABORATIVE,
    "comprehensive": PIPELINE_COMPREHENSIVE,
    "shuttle_optimized": PIPELINE_SHUTTLE_OPTIMIZED,
}


def get_pipeline(name: str) -> PipelineDefinition:
    """
    Get pipeline by name.

    Args:
        name: Pipeline name (simple, verified, quality_assured, etc.)

    Returns:
        PipelineDefinition

    Raises:
        KeyError: If pipeline name not found
    """
    if name not in PIPELINES:
        available = ", ".join(PIPELINES.keys())
        raise KeyError(f"Pipeline '{name}' not found. Available: {available}")
    return PIPELINES[name]


def list_pipelines() -> list:
    """List all available pipeline names."""
    return list(PIPELINES.keys())


def get_pipeline_info(name: str) -> dict:
    """
    Get pipeline metadata.

    Args:
        name: Pipeline name

    Returns:
        dict with description, target_latency_ms, use_case
    """
    pipeline = get_pipeline(name)
    return pipeline.metadata