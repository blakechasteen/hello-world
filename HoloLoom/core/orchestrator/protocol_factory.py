#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocol-Based Pipeline Factory - Dependency Injection for Orchestrator
========================================================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 3: Protocol DI (Days 11-14)

Provides factory functions for creating orchestrator components using
protocol-based dependency injection. Enables swapping implementations
without changing orchestrator code.

Key Components:
    - OrchestratorComponents: Dataclass container for all protocol components
    - create_component_defaults: Factory for default implementations
    - create_orchestrator_from_protocols: Main factory with DI support
    - create_pipeline_with_protocols: Pipeline factory using protocols

Author: Claude Code (Elegance Pass - Phase 3)
Date: 2025-12-11
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.config import Config
    from HoloLoom.loom.command import LoomCommand
    from HoloLoom.memory.graph import KG
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
    from HoloLoom.tools.executor import ToolExecutor
    from HoloLoom.alignment.safety_guardrails import SafetyGuardrails
    from HoloLoom.alignment.audit_trail import AuditTrail

from HoloLoom.orchestrator.protocols import (
    PatternSelectorProtocol,
    ThreadSelectorProtocol,
    FeatureExtractorProtocol,
    ConvergenceProtocol,
    ToolExecutorProtocol,
    SpacetimeAssemblerProtocol,
    DefaultPatternSelector,
    DefaultThreadSelector,
    DefaultFeatureExtractor,
    DefaultConvergence,
    DefaultToolExecutor,
    DefaultSpacetimeAssembler,
)
from HoloLoom.orchestrator.pipeline import ExecutorPipeline

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorComponents:
    """
    Container for all protocol components used by the orchestrator.

    Holds instances of each protocol implementation, enabling dependency
    injection and easy swapping of components without changing orchestrator code.

    Attributes:
        pattern_selector: Selects processing patterns (BARE/FAST/FUSED)
        thread_selector: Selects memory threads from Yarn Graph
        feature_extractor: Extracts features via Resonance Shed
        convergence: Collapses decisions via Thompson Sampling
        tool_executor: Executes selected tools with safety gating
        spacetime_assembler: Assembles final Spacetime output

    Example:
        >>> components = OrchestratorComponents(
        ...     pattern_selector=DefaultPatternSelector(loom_command=loom_cmd),
        ...     thread_selector=DefaultThreadSelector(yarn_graph=kg),
        ...     feature_extractor=DefaultFeatureExtractor(embedder=embedder, cfg=config),
        ...     convergence=DefaultConvergence(engine=conv_engine, n_tools=4),
        ...     tool_executor=DefaultToolExecutor(tool_executor=tool_exec),
        ...     spacetime_assembler=DefaultSpacetimeAssembler(cfg=config),
        ... )
        >>> assert components.is_complete()
        >>> orchestrator = create_orchestrator_from_protocols(components, cfg=config)
    """

    pattern_selector: Optional[PatternSelectorProtocol] = None
    thread_selector: Optional[ThreadSelectorProtocol] = None
    feature_extractor: Optional[FeatureExtractorProtocol] = None
    convergence: Optional[ConvergenceProtocol] = None
    tool_executor: Optional[ToolExecutorProtocol] = None
    spacetime_assembler: Optional[SpacetimeAssemblerProtocol] = None

    def is_complete(self) -> bool:
        """
        Check if all required components are set.

        Returns:
            True if all 6 components are non-None
        """
        return all([
            self.pattern_selector is not None,
            self.thread_selector is not None,
            self.feature_extractor is not None,
            self.convergence is not None,
            self.tool_executor is not None,
            self.spacetime_assembler is not None,
        ])

    def missing_components(self) -> List[str]:
        """
        Get list of missing component names.

        Returns:
            List of component attribute names that are None
        """
        missing = []
        if self.pattern_selector is None:
            missing.append('pattern_selector')
        if self.thread_selector is None:
            missing.append('thread_selector')
        if self.feature_extractor is None:
            missing.append('feature_extractor')
        if self.convergence is None:
            missing.append('convergence')
        if self.tool_executor is None:
            missing.append('tool_executor')
        if self.spacetime_assembler is None:
            missing.append('spacetime_assembler')
        return missing

    def validate(self) -> None:
        """
        Validate that all components are set.

        Raises:
            ValueError: If any components are missing
        """
        missing = self.missing_components()
        if missing:
            raise ValueError(
                f"OrchestratorComponents incomplete. Missing: {', '.join(missing)}"
            )


def create_component_defaults(
    cfg: 'Config',
    loom_command: 'LoomCommand',
    yarn_graph: 'KG',
    embedder: 'MatryoshkaEmbeddings',
    policy: Any,
    tool_executor: 'ToolExecutor',
    memory: Optional[Any] = None,
    retriever: Optional[Any] = None,
    guardrails: Optional['SafetyGuardrails'] = None,
    audit_trail: Optional['AuditTrail'] = None,
    semantic_cache: Optional[Any] = None,
    dashboard_constructor: Optional[Any] = None,
    awareness_context: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> OrchestratorComponents:
    """
    Factory for creating default protocol implementations.

    Creates all 6 default implementations using the provided dependencies.
    This is the standard way to bootstrap an orchestrator with default behavior.

    Args:
        cfg: Configuration object
        loom_command: LoomCommand for pattern selection
        yarn_graph: KG instance for thread selection
        embedder: MatryoshkaEmbeddings for multi-scale embeddings
        policy: Policy engine for neural predictions
        tool_executor: ToolExecutor for running tools
        memory: Optional dynamic memory backend
        retriever: Optional legacy retriever
        guardrails: Optional SafetyGuardrails for action gating
        audit_trail: Optional AuditTrail for logging decisions
        semantic_cache: Optional semantic cache for statistics
        dashboard_constructor: Optional dashboard constructor
        awareness_context: Optional awareness/consciousness context
        logger: Optional logger instance

    Returns:
        OrchestratorComponents with all 6 defaults populated

    Example:
        >>> components = create_component_defaults(
        ...     cfg=config,
        ...     loom_command=loom_cmd,
        ...     yarn_graph=kg,
        ...     embedder=embedder,
        ...     policy=policy,
        ...     tool_executor=tool_exec,
        ... )
        >>> assert components.is_complete()
    """
    log = logger or logging.getLogger(__name__)

    # Build convergence engine from policy if available
    convergence_engine = None
    if policy is not None:
        try:
            from HoloLoom.convergence.engine import ConvergenceEngine
            convergence_engine = ConvergenceEngine(policy=policy, cfg=cfg)
        except (ImportError, Exception) as e:
            log.debug(f"Could not create ConvergenceEngine: {e}")

    return OrchestratorComponents(
        pattern_selector=DefaultPatternSelector(
            loom_command=loom_command,
            logger=log,
        ),
        thread_selector=DefaultThreadSelector(
            yarn_graph=yarn_graph,
            logger=log,
        ),
        feature_extractor=DefaultFeatureExtractor(
            embedder=embedder,
            cfg=cfg,
            logger=log,
        ),
        convergence=DefaultConvergence(
            engine=convergence_engine,
            n_tools=getattr(cfg, 'n_tools', 4),
            epsilon=getattr(cfg, 'epsilon', 0.1),
            logger=log,
        ),
        tool_executor=DefaultToolExecutor(
            tool_executor=tool_executor,
            guardrails=guardrails,
            logger=log,
        ),
        spacetime_assembler=DefaultSpacetimeAssembler(
            cfg=cfg,
            semantic_cache=semantic_cache,
            logger=log,
        ),
    )


def create_orchestrator_from_protocols(
    components: OrchestratorComponents,
    cfg: 'Config',
    enable_shuttle: bool = False,
    shuttle_stage: Optional[Any] = None,
    gradient_router: Optional[Any] = None,
    multipass_memory_crawl: Optional[Callable] = None,
    linguistic_gate: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    emit_stage_event: Optional[Callable[[int, str, float], None]] = None,
) -> 'ProtocolOrchestrator':
    """
    Create an orchestrator using protocol-based components.

    This factory enables full dependency injection: any component can be
    swapped by providing an alternative implementation of its protocol.

    Args:
        components: OrchestratorComponents with all protocol implementations
        cfg: Configuration object
        enable_shuttle: Whether to use Shuttle for thread selection
        shuttle_stage: Optional ShuttleStage for advanced selection
        gradient_router: Optional gradient flow router
        multipass_memory_crawl: Optional async function for multipass crawling
        linguistic_gate: Optional linguistic gate for Phase 5
        logger: Optional logger instance
        emit_stage_event: Optional callback for stage monitoring

    Returns:
        ProtocolOrchestrator configured with provided components

    Raises:
        ValueError: If components is incomplete

    Example:
        >>> # Use defaults
        >>> components = create_component_defaults(cfg, loom_cmd, kg, embedder, policy, tool_exec)
        >>> orchestrator = create_orchestrator_from_protocols(components, cfg)
        >>>
        >>> # Swap one component
        >>> components.convergence = MyCustomConvergence(cfg, policy)
        >>> orchestrator = create_orchestrator_from_protocols(components, cfg)
    """
    components.validate()
    log = logger or logging.getLogger(__name__)

    return ProtocolOrchestrator(
        components=components,
        cfg=cfg,
        enable_shuttle=enable_shuttle,
        shuttle_stage=shuttle_stage,
        gradient_router=gradient_router,
        multipass_memory_crawl=multipass_memory_crawl,
        linguistic_gate=linguistic_gate,
        logger=log,
        emit_stage_event=emit_stage_event,
    )


class ProtocolOrchestrator:
    """
    Orchestrator that uses protocol-based components.

    This orchestrator delegates all work to injected protocol implementations,
    enabling full customization without modifying orchestrator code.

    Attributes:
        components: OrchestratorComponents container
        cfg: Configuration object
        enable_shuttle: Whether Shuttle is enabled
        shuttle_stage: Optional ShuttleStage
        gradient_router: Optional gradient router
        multipass_memory_crawl: Optional multipass crawler
        linguistic_gate: Optional linguistic gate
        logger: Logger instance

    Example:
        >>> orchestrator = create_orchestrator_from_protocols(components, cfg)
        >>> ctx = await orchestrator.orchestrate(ctx)
    """

    def __init__(
        self,
        components: OrchestratorComponents,
        cfg: 'Config',
        enable_shuttle: bool = False,
        shuttle_stage: Optional[Any] = None,
        gradient_router: Optional[Any] = None,
        multipass_memory_crawl: Optional[Callable] = None,
        linguistic_gate: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        emit_stage_event: Optional[Callable[[int, str, float], None]] = None,
    ):
        """Initialize the protocol orchestrator."""
        self.components = components
        self.cfg = cfg
        self.enable_shuttle = enable_shuttle
        self.shuttle_stage = shuttle_stage
        self.gradient_router = gradient_router
        self.multipass_memory_crawl = multipass_memory_crawl
        self.linguistic_gate = linguistic_gate
        self.logger = logger or logging.getLogger(__name__)
        self.emit_stage_event = emit_stage_event

    async def orchestrate(self, ctx: 'WeavingContext') -> 'WeavingContext':
        """
        Execute the full weaving cycle using protocol components.

        Delegates to each protocol component in sequence:
        1. Pattern selection
        2. Thread selection
        3. Feature extraction
        4. Convergence (decision)
        5. Tool execution
        6. Spacetime assembly

        Args:
            ctx: Initial WeavingContext

        Returns:
            Final WeavingContext after all stages complete
        """
        import numpy as np

        # Extract query text for protocols that need string input
        query_text = ctx.query.text if hasattr(ctx.query, 'text') else str(ctx.query)

        # Step 1: Pattern Selection
        self._emit_event(1, "pattern_selection_start")
        pattern = self.components.pattern_selector.select_pattern(query_text)
        ctx = ctx.with_pattern(pattern)
        self._emit_event(1, "pattern_selection_complete")

        # Step 3: Thread Selection
        self._emit_event(3, "thread_selection_start")
        threads = await self.components.thread_selector.select_threads(
            ctx.chrono_trigger,  # temporal_window first
            ctx.query,           # query second
            limit=10,
        )
        ctx = ctx.with_threads(threads)
        self._emit_event(3, "thread_selection_complete")

        # Steps 4-6: Feature Extraction
        self._emit_event(4, "feature_extraction_start")
        features = await self.components.feature_extractor.extract(
            ctx.query,
            ctx.threads,
        )
        ctx = ctx.with_features(features)
        self._emit_event(6, "feature_extraction_complete")

        # Step 7: Convergence
        self._emit_event(7, "convergence_start")
        # Get probability distribution from features (or use uniform if not available)
        embeddings = features.get('embeddings', None)
        if embeddings is not None:
            # Use embeddings to compute tool probabilities (simplified)
            n_tools = getattr(self.cfg, 'n_tools', 4)
            probs = np.ones(n_tools) / n_tools  # Uniform distribution as placeholder
        else:
            probs = np.ones(4) / 4

        collapse_result = self.components.convergence.collapse(
            probs,
            strategy=getattr(self.cfg, 'collapse_strategy', 'bayesian_blend'),
        )
        ctx = ctx.with_decision(collapse_result)
        self._emit_event(7, "convergence_complete")

        # Step 8: Tool Execution
        self._emit_event(8, "tool_execution_start")
        # Map tool index to tool name
        tools = self.components.tool_executor.get_available_tools()
        tool_index = collapse_result.tool_index if hasattr(collapse_result, 'tool_index') else 0
        tool_name = tools[tool_index] if tool_index < len(tools) else tools[0]

        tool_result = await self.components.tool_executor.execute(
            tool_name,
            ctx.query,
            {'features': features, 'threads': ctx.threads},
        )
        ctx = ctx.with_tool_result(tool_result)
        self._emit_event(8, "tool_execution_complete")

        # Step 9: Spacetime Assembly
        self._emit_event(9, "spacetime_assembly_start")
        spacetime = self.components.spacetime_assembler.assemble(ctx)
        ctx = ctx.with_spacetime(spacetime)
        self._emit_event(9, "spacetime_assembly_complete")

        return ctx

    def _emit_event(self, stage_id: int, event_name: str, duration: float = 0.0) -> None:
        """Emit stage event if callback registered."""
        if self.emit_stage_event:
            self.emit_stage_event(stage_id, event_name, duration)


def create_pipeline_with_protocols(
    components: OrchestratorComponents,
    cfg: 'Config',
    enable_shuttle: bool = False,
    shuttle_stage: Optional[Any] = None,
    gradient_router: Optional[Any] = None,
    multipass_memory_crawl: Optional[Callable] = None,
    linguistic_gate: Optional[Any] = None,
    guardrails: Optional['SafetyGuardrails'] = None,
    audit_trail: Optional['AuditTrail'] = None,
    logger: Optional[logging.Logger] = None,
    emit_stage_event: Optional[Callable[[int, str, float], None]] = None,
) -> ExecutorPipeline:
    """
    Create an ExecutorPipeline using protocol-based components.

    This factory creates stage executors that delegate to the provided
    protocol implementations, enabling component swapping.

    Args:
        components: OrchestratorComponents with protocol implementations
        cfg: Configuration object
        enable_shuttle: Whether to use Shuttle for thread selection
        shuttle_stage: Optional ShuttleStage
        gradient_router: Optional gradient router
        multipass_memory_crawl: Optional multipass crawler
        linguistic_gate: Optional linguistic gate
        guardrails: Optional SafetyGuardrails
        audit_trail: Optional AuditTrail
        logger: Optional logger
        emit_stage_event: Optional stage event callback

    Returns:
        ExecutorPipeline with protocol-aware executors

    Raises:
        ValueError: If components is incomplete

    Example:
        >>> components = create_component_defaults(cfg, loom_cmd, kg, embedder, policy, tool_exec)
        >>> # Swap convergence with custom implementation
        >>> components.convergence = MyCustomConvergence(cfg, policy)
        >>> pipeline = create_pipeline_with_protocols(components, cfg)
        >>> ctx = await pipeline.execute(ctx)
    """
    components.validate()
    log = logger or logging.getLogger(__name__)

    from HoloLoom.orchestrator.stages.executors import (
        MetaPromptExecutor,
        PatternSelectionExecutor,
        ChronoTriggerExecutor,
        ThreadSelectionExecutor,
        ParallelFeatureExecutor,
        ConvergenceExecutor,
        ToolExecutionExecutor,
        SpacetimeExecutor,
    )

    # Extract underlying dependencies from default implementations
    # This allows mixing protocol components with standard executors
    loom_command = getattr(components.pattern_selector, 'loom_command', None)
    yarn_graph = getattr(components.thread_selector, 'yarn_graph', None)
    embedder = getattr(components.feature_extractor, 'embedder', None)
    resonance_shed = getattr(components.feature_extractor, 'resonance_shed', None)
    engine = getattr(components.convergence, 'engine', None)
    # Try to get policy from engine if available
    policy = getattr(engine, 'policy', None) if engine else None
    tool_executor = getattr(components.tool_executor, 'tool_executor', None)
    semantic_cache = getattr(components.spacetime_assembler, 'semantic_cache', None)
    # For backward compatibility, we don't extract memory/retriever from defaults
    # as they are not part of the simplified default implementation
    memory = None
    retriever = None
    # These are optional in SpacetimeExecutor but not in DefaultSpacetimeAssembler
    dashboard_constructor = None
    awareness_context = None

    executors = [
        # Step 0: Meta-Prompt Enhancement (disabled by default)
        MetaPromptExecutor(
            enable_enhancement=getattr(cfg, 'enable_meta_prompt', False),
            proto_llm_call=None,
            logger=log,
            emit_stage_event=emit_stage_event,
        ),

        # Step 1: Pattern Selection
        PatternSelectionExecutor(
            loom_command=loom_command,
            logger=log,
            emit_stage_event=emit_stage_event,
        ),

        # Step 2: Chrono Trigger
        ChronoTriggerExecutor(
            lookback_days=getattr(cfg, 'lookback_days', 365),
            recency_bias=getattr(cfg, 'recency_bias', 0.5),
            logger=log,
            emit_stage_event=emit_stage_event,
        ),

        # Step 3: Thread Selection
        ThreadSelectionExecutor(
            yarn_graph=yarn_graph,
            shuttle_stage=shuttle_stage,
            enable_shuttle=enable_shuttle,
            logger=log,
            emit_stage_event=emit_stage_event,
        ),

        # Steps 4-6: Parallel Feature Pipeline
        ParallelFeatureExecutor(
            cfg=cfg,
            embedder=embedder,
            memory=memory,
            retriever=retriever,
            linguistic_gate=linguistic_gate,
            guardrails=guardrails,
            multipass_memory_crawl=multipass_memory_crawl,
            logger=log,
            emit_stage_event=emit_stage_event,
        ),

        # Step 7: Convergence Engine
        ConvergenceExecutor(
            cfg=cfg,
            policy=policy,
            tool_executor=tool_executor,
            gradient_router=gradient_router,
            logger=log,
            emit_stage_event=emit_stage_event,
        ),

        # Step 8: Tool Execution
        ToolExecutionExecutor(
            tool_executor=tool_executor,
            guardrails=guardrails,
            audit_trail=audit_trail,
            logger=log,
            emit_stage_event=emit_stage_event,
        ),

        # Step 9: Spacetime Fabric
        SpacetimeExecutor(
            cfg=cfg,
            semantic_cache=semantic_cache,
            dashboard_constructor=dashboard_constructor,
            awareness_context=awareness_context,
            logger=log,
            emit_stage_event=emit_stage_event,
        ),
    ]

    return ExecutorPipeline(executors)


__all__ = [
    'OrchestratorComponents',
    'create_component_defaults',
    'create_orchestrator_from_protocols',
    'create_pipeline_with_protocols',
    'ProtocolOrchestrator',
]
