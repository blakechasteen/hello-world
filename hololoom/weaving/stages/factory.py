"""
Stage Factory
=============

Builds all stage instances from a Config + components dict.
Graceful degradation: if a required component is missing,
that stage is excluded with a warning — not a crash.

Created in PR 8 (A-8).
"""

import logging
from typing import Any

from hololoom.weaving.stages import (
    BetaWavePackingStage,
    ConscienceGatingStage,
    ConsciousnessPerceptionStage,
    DecisionCollapseStage,
    FabricWeavingStage,
    FastExecutionStage,
    FeatureExtractionStage,
    JennyCompilationStage,
    MemoryRetrievalStage,
    # Optional (7)
    MetaPromptEnhancementStage,
    # Core (11)
    PatternSelectionStage,
    ProductionMetricsStage,
    # Lightweight (3)
    QuickDecisionStage,
    RecursiveLearningStage,
    ReflectionStage,
    SmartRoutingStage,
    TemporalControlStage,
    ThreadSelectionStage,
    ToolExecutionStage,
    WarpComputeStage,
    WarpTensioningStage,
)

logger = logging.getLogger(__name__)


def _get(components: dict, key: str, default=None):
    """Get a component, returning default if missing or None."""
    return components.get(key, default)


def build_stages(
    config: Any,
    components: dict[str, Any],
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    """Build all stage instances whose dependencies are available.

    Args:
        config: HoloLoom Config instance. Used for stage parameters
            (token budgets, thresholds, etc.) and sub-config access.
        components: Dict of initialized component instances. Keys:
            - loom_command: LoomCommand for pattern selection
            - chrono_trigger: ChronoTrigger for temporal control
            - yarn_graph: Memory graph for thread selection
            - shuttle_stage: Optional MCTS shuttle
            - warp_space: WarpSpace for tensioning/compute/weaving
            - memory_backend: UnifiedMemory or compatible
            - retriever: Legacy shard retriever
            - policy_engine: Policy for decision collapse
            - convergence_engine: ConvergenceEngine
            - tool_executor: Tool execution engine
            - guardrails: SafetyGuardrails
            - audit_trail: AuditTrail
            - semantic_cache: Semantic spectrum cache
            - linguistic_gate: Linguistic Matryoshka gate
            - spring_engine: Spring activation engine
            - awareness_graph: Consciousness awareness graph
            - evaluator: Conscience ethical evaluator
            - classifier: Smart routing classifier
            - reflection_buffer: Reflection episodic buffer
            - learning_loop: Recursive learning loop
            - semantic_bandit: Thompson sampling bandit
            - jenny_runtime: Jenny panel compilation runtime
            - metrics_collector: Prometheus metrics collector
            - proto_llm_call: LLM callable for meta-prompt enhancement
            - semantic_state_class: Semantic state for warp compute
            - semantic_tool_selector: Tool selector for warp compute
        log: Optional logger (uses module logger if not provided).

    Returns:
        Dict mapping stage_name → stage instance. Only includes stages
        whose required dependencies are satisfied.
    """
    log = log or logger
    stages: dict[str, Any] = {}

    # =====================================================================
    # Stage builders: each tries to construct its stage.
    # Returns (name, instance) or None if deps missing.
    # =====================================================================

    def _try_build(name: str, builder, required_keys=None):
        """Try to build a stage, warning if required deps are missing."""
        if required_keys:
            missing = [k for k in required_keys if not _get(components, k)]
            if missing:
                log.debug(
                    f"  Stage '{name}' excluded: missing {missing}"
                )
                return
        try:
            instance = builder()
            stages[name] = instance
        except Exception as e:
            log.warning(f"  Stage '{name}' failed to build: {e}")

    # ----- Core stages -----

    _try_build('pattern_selection', lambda: PatternSelectionStage(
        loom_command=components['loom_command'],
        default_pattern=_get(components, 'default_pattern'),
        enable_auto_detect=getattr(config, 'enable_complexity_auto_detect', True),
        logger=log,
    ), required_keys=['loom_command'])

    _try_build('temporal_control', lambda: TemporalControlStage(
        chrono_trigger=components['chrono_trigger'],
        logger=log,
    ), required_keys=['chrono_trigger'])

    _try_build('thread_selection', lambda: ThreadSelectionStage(
        yarn_graph=components['yarn_graph'],
        shuttle_stage=_get(components, 'shuttle_stage'),
        enable_shuttle=bool(_get(components, 'shuttle_stage')),
        logger=log,
    ), required_keys=['yarn_graph'])

    _try_build('feature_extraction', lambda: FeatureExtractionStage(
        config=config,
        linguistic_gate=_get(components, 'linguistic_gate'),
        guardrails=_get(components, 'guardrails'),
        logger=log,
    ))

    _try_build('warp_tensioning', lambda: WarpTensioningStage(
        warp_space=components['warp_space'],
        logger=log,
    ), required_keys=['warp_space'])

    _try_build('warp_compute', lambda: WarpComputeStage(
        warp_space=components['warp_space'],
        semantic_state_class=_get(components, 'semantic_state_class'),
        semantic_tool_selector=_get(components, 'semantic_tool_selector'),
        logger=log,
    ), required_keys=['warp_space'])

    _try_build('memory_retrieval', lambda: MemoryRetrievalStage(
        memory_backend=_get(components, 'memory_backend'),
        retriever=_get(components, 'retriever'),
        logger=log,
    ))

    _try_build('beta_wave_packing', lambda: BetaWavePackingStage(
        spring_engine=_get(components, 'spring_engine'),
        token_budget=getattr(
            getattr(config, 'beta_wave_packing', None), 'token_budget', 4096),
        query_reserve=getattr(
            getattr(config, 'beta_wave_packing', None), 'query_reserve', 512),
        response_reserve=getattr(
            getattr(config, 'beta_wave_packing', None), 'response_reserve', 1024),
        activation_threshold=getattr(
            getattr(config, 'beta_wave_packing', None), 'activation_threshold', 0.1),
        compression_threshold=getattr(
            getattr(config, 'beta_wave_packing', None), 'compression_threshold', 0.3),
        logger=log,
    ))

    _try_build('decision_collapse', lambda: DecisionCollapseStage(
        policy_engine=components['policy_engine'],
        convergence_engine=components['convergence_engine'],
        logger=log,
    ), required_keys=['policy_engine', 'convergence_engine'])

    _try_build('tool_execution', lambda: ToolExecutionStage(
        tool_executor=components['tool_executor'],
        guardrails=_get(components, 'guardrails'),
        audit_trail=_get(components, 'audit_trail'),
        logger=log,
    ), required_keys=['tool_executor'])

    _try_build('fabric_weaving', lambda: FabricWeavingStage(
        warp_space=_get(components, 'warp_space'),
        semantic_cache=_get(components, 'semantic_cache'),
        logger=log,
    ))

    # ----- Lightweight stages -----

    _try_build('quick_decision', lambda: QuickDecisionStage(
        tool_executor=components['tool_executor'],
        routing_table=_get(components, 'routing_table'),
        default_tool=getattr(config, 'default_tool', 'knowledge_base'),
        logger=log,
    ), required_keys=['tool_executor'])

    _try_build('fast_execution', lambda: FastExecutionStage(
        logger=log,
    ))

    _try_build('reflection', lambda: ReflectionStage(
        reflection_buffer=_get(components, 'reflection_buffer'),
        logger=log,
    ))

    # ----- Optional stages -----

    _try_build('meta_prompt_enhancement', lambda: MetaPromptEnhancementStage(
        proto_llm_call=_get(components, 'proto_llm_call'),
        enable_enhancement=getattr(config, 'auto_enhance', True),
        logger=log,
    ))

    _try_build('smart_routing', lambda: SmartRoutingStage(
        classifier=_get(components, 'classifier'),
        fast_path_recipes=_get(components, 'fast_path_recipes'),
        logger=log,
    ))

    _try_build('consciousness_perception', lambda: ConsciousnessPerceptionStage(
        awareness_graph=_get(components, 'awareness_graph'),
        logger=log,
    ))

    _try_build('conscience_gating', lambda: ConscienceGatingStage(
        evaluator=_get(components, 'evaluator'),
        logger=log,
    ))

    _try_build('jenny_compilation', lambda: JennyCompilationStage(
        jenny_runtime=_get(components, 'jenny_runtime'),
        panel_count=getattr(
            getattr(config, 'jenny', None), 'max_panels_per_query', 4),
        logger=log,
    ))

    _try_build('recursive_learning', lambda: RecursiveLearningStage(
        learning_loop=_get(components, 'learning_loop'),
        semantic_bandit=_get(components, 'semantic_bandit'),
        success_threshold=getattr(
            getattr(config, 'semantic_bandit', None), 'success_threshold', 0.5),
        logger=log,
    ))

    _try_build('production_metrics', lambda: ProductionMetricsStage(
        metrics_collector=_get(components, 'metrics_collector'),
        logger=log,
    ))

    log.info(f"  Factory built {len(stages)} stages: {sorted(stages.keys())}")
    return stages
