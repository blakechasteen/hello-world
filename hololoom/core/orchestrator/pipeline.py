#!/usr/bin/env python3
"""
Executor Pipeline - Chains Stage Executors into a Pipeline
============================================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 2: Stage Executor Classes (Day 10)

Provides ExecutorPipeline for chaining stage executors into a complete
weaving pipeline, plus factory functions for creating standard pipelines.

Key Components:
    - ExecutorPipeline: Chains executors sequentially
    - create_default_pipeline: Factory for standard 8-executor pipeline
    - create_minimal_pipeline: Factory for minimal 4-executor pipeline

Author: Claude Code (Elegance Pass - Phase 2)
Date: 2025-12-09
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hololoom.alignment.audit_trail import AuditTrail
    from hololoom.alignment.safety_guardrails import SafetyGuardrails
    from hololoom.config import Config
    from hololoom.embedding.spectral import MatryoshkaEmbeddings
    from hololoom.loom.command import LoomCommand
    from hololoom.memory.graph import KG
    from hololoom.orchestrator.context import WeavingContext
    from hololoom.orchestrator.protocols import StageExecutorProtocol
    from hololoom.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class ExecutorPipeline:
    """
    Chains stage executors into a pipeline.

    The pipeline executes all stages in sequence, passing the WeavingContext
    through each executor. Provides iteration, length, and indexing support.

    Attributes:
        executors: List of stage executors in order

    Example:
        >>> pipeline = ExecutorPipeline([
        ...     PatternSelectionExecutor(loom_command),
        ...     ChronoTriggerExecutor(),
        ...     ThreadSelectionExecutor(yarn_graph),
        ...     ParallelFeatureExecutor(cfg, embedder),
        ...     ConvergenceExecutor(cfg, policy, tool_executor),
        ...     ToolExecutionExecutor(tool_executor),
        ...     SpacetimeExecutor(cfg),
        ... ])
        >>> ctx = await pipeline.execute(ctx)
        >>> print(f"Pipeline complete: {ctx.spacetime.response}")

    Design Notes:
        - Executors are executed in order
        - Each executor receives the WeavingContext from the previous stage
        - Failed stages raise exceptions (no automatic error recovery)
        - Consider wrapping in try/except for graceful error handling
    """

    def __init__(self, executors: list[StageExecutorProtocol]):
        """
        Initialize the executor pipeline.

        Args:
            executors: List of stage executors to chain together
        """
        self.executors = executors

    async def execute(self, ctx: WeavingContext) -> WeavingContext:
        """
        Execute all stages in sequence.

        Args:
            ctx: Initial WeavingContext

        Returns:
            Final WeavingContext after all stages complete

        Raises:
            Exception: If any stage fails
        """
        for executor in self.executors:
            ctx = await executor.execute(ctx)
        return ctx

    def __len__(self) -> int:
        """Return number of executors in pipeline."""
        return len(self.executors)

    def __iter__(self):
        """Iterate over executors."""
        return iter(self.executors)

    def __getitem__(self, index: int) -> StageExecutorProtocol:
        """Get executor by index."""
        return self.executors[index]

    def stage_ids(self) -> list[int]:
        """Return list of stage IDs in order."""
        return [e.stage_id for e in self.executors]

    def stage_names(self) -> list[str]:
        """Return list of stage names in order."""
        return [e.stage_name for e in self.executors]


def create_default_pipeline(
    cfg: Config,
    loom_command: LoomCommand,
    yarn_graph: KG,
    embedder: MatryoshkaEmbeddings,
    policy: Any,
    tool_executor: ToolExecutor,
    memory: Any | None = None,
    retriever: Any | None = None,
    linguistic_gate: Any | None = None,
    guardrails: SafetyGuardrails | None = None,
    audit_trail: AuditTrail | None = None,
    semantic_cache: Any | None = None,
    dashboard_constructor: Any | None = None,
    multipass_memory_crawl: Callable | None = None,
    enable_shuttle: bool = False,
    shuttle_stage: Any | None = None,
    gradient_router: Any | None = None,
    awareness_context: dict | None = None,
    logger: logging.Logger | None = None,
    emit_stage_event: Callable[[int, str, float], None] | None = None,
) -> ExecutorPipeline:
    """
    Factory for creating a standard 8-executor weaving pipeline.

    Creates a complete pipeline covering all 9 weaving steps:
    - Step 0: Meta-Prompt Enhancement (disabled by default)
    - Step 1: Pattern Selection (Loom Command)
    - Step 2: Chrono Trigger (Temporal Window)
    - Step 3: Thread Selection (Yarn Graph / Shuttle)
    - Steps 4-6: Parallel Feature Pipeline (Resonance + Warp + Retrieval)
    - Step 7: Convergence Engine (Decision Collapse)
    - Step 8: Tool Execution (Safety-Gated)
    - Step 9: Spacetime Fabric (Result Assembly)

    Args:
        cfg: Configuration object
        loom_command: LoomCommand for pattern selection
        yarn_graph: KG instance for thread selection
        embedder: MatryoshkaEmbeddings for multi-scale embeddings
        policy: Policy engine for neural predictions
        tool_executor: ToolExecutor for running tools
        memory: Optional dynamic memory backend
        retriever: Optional legacy retriever
        linguistic_gate: Optional linguistic gate for Phase 5
        guardrails: Optional SafetyGuardrails for action gating
        audit_trail: Optional AuditTrail for logging decisions
        semantic_cache: Optional semantic cache for statistics
        dashboard_constructor: Optional dashboard constructor
        multipass_memory_crawl: Optional async function for multipass crawling
        enable_shuttle: Whether to use Shuttle for thread selection
        shuttle_stage: Optional ShuttleStage for advanced selection
        gradient_router: Optional gradient flow router
        awareness_context: Optional awareness/consciousness context
        logger: Optional logger instance
        emit_stage_event: Optional callback for stage monitoring

    Returns:
        ExecutorPipeline with 8 executors (Steps 0-9, with Steps 4-6 combined)

    Example:
        >>> pipeline = create_default_pipeline(
        ...     cfg=config,
        ...     loom_command=loom_command,
        ...     yarn_graph=kg,
        ...     embedder=embedder,
        ...     policy=policy,
        ...     tool_executor=tool_executor,
        ... )
        >>> ctx = await pipeline.execute(ctx)
    """
    from hololoom.orchestrator.stages.executors import (
        ChronoTriggerExecutor,
        ConvergenceExecutor,
        MetaPromptExecutor,
        ParallelFeatureExecutor,
        PatternSelectionExecutor,
        SpacetimeExecutor,
        ThreadSelectionExecutor,
        ToolExecutionExecutor,
    )

    log = logger or logging.getLogger(__name__)

    # Create all 8 executors
    executors = [
        # Step 0: Meta-Prompt Enhancement (disabled by default)
        MetaPromptExecutor(
            enable_enhancement=getattr(cfg, 'enable_meta_prompt', False),
            proto_llm_call=None,  # Would need LLM client
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


def create_minimal_pipeline(
    cfg: Config,
    loom_command: LoomCommand,
    yarn_graph: KG,
    embedder: MatryoshkaEmbeddings,
    policy: Any,
    tool_executor: ToolExecutor,
    logger: logging.Logger | None = None,
    emit_stage_event: Callable[[int, str, float], None] | None = None,
) -> ExecutorPipeline:
    """
    Factory for creating a minimal 5-executor pipeline.

    Creates a stripped-down pipeline for fast execution:
    - Step 1: Pattern Selection
    - Step 2: Chrono Trigger
    - Steps 4-6: Parallel Feature Pipeline
    - Step 7: Convergence Engine
    - Step 9: Spacetime Fabric

    Skips:
    - Step 0 (Meta-Prompt Enhancement)
    - Step 3 (Thread Selection - uses simple shards)
    - Step 8 (Tool Execution - safety gating)

    Args:
        cfg: Configuration object
        loom_command: LoomCommand for pattern selection
        yarn_graph: KG instance (minimal use)
        embedder: MatryoshkaEmbeddings
        policy: Policy engine
        tool_executor: ToolExecutor
        logger: Optional logger
        emit_stage_event: Optional callback

    Returns:
        ExecutorPipeline with 5 executors
    """
    from hololoom.orchestrator.stages.executors import (
        ChronoTriggerExecutor,
        ConvergenceExecutor,
        ParallelFeatureExecutor,
        PatternSelectionExecutor,
        SpacetimeExecutor,
    )

    log = logger or logging.getLogger(__name__)

    executors = [
        PatternSelectionExecutor(loom_command=loom_command, logger=log, emit_stage_event=emit_stage_event),
        ChronoTriggerExecutor(logger=log, emit_stage_event=emit_stage_event),
        ParallelFeatureExecutor(cfg=cfg, embedder=embedder, logger=log, emit_stage_event=emit_stage_event),
        ConvergenceExecutor(cfg=cfg, policy=policy, tool_executor=tool_executor, logger=log, emit_stage_event=emit_stage_event),
        SpacetimeExecutor(cfg=cfg, logger=log, emit_stage_event=emit_stage_event),
    ]

    return ExecutorPipeline(executors)


__all__ = [
    'ExecutorPipeline',
    'create_default_pipeline',
    'create_minimal_pipeline',
]
