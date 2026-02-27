#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Feature Executor - Steps 4-6: Resonance + Warp + Retrieval
====================================================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 2: Stage Executor Classes

Executor class that wraps the parallelized Steps 4-6:
- Step 4: Resonance Shed (DotPlasma creation via feature threads)
- Step 5: Warp Space (tension threads into continuous manifold)
- Step 6: Memory Retrieval (multipass crawl or legacy retrieval)
- Step 5.5: Warp Compute (tensor operations in continuous space)
- Step 6.5: Beta Wave Context Packing (physics-based optimization)

These steps execute concurrently using asyncio.gather for 40-120ms speedup.

Author: Claude Code (Elegance Pass - Phase 2)
Date: 2025-12-09
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Any, TYPE_CHECKING

from HoloLoom.orchestrator.protocols import BaseStageExecutor

if TYPE_CHECKING:
    from HoloLoom.orchestrator.context import WeavingContext
    from HoloLoom.config import Config
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
    from HoloLoom.alignment.safety_guardrails import SafetyGuardrails


class ParallelFeatureExecutor(BaseStageExecutor):
    """
    Steps 4-6: Parallelized Feature Extraction Pipeline.

    Executes three independent steps concurrently for 40-120ms speedup:
    - Step 4: Feature extraction through Resonance Shed (DotPlasma)
    - Step 5: Warp Space tensioning
    - Step 6: Memory retrieval

    Also handles post-parallel steps:
    - Step 5.5: Warp compute (tensor operations)
    - Step 6.5: Beta wave context packing (physics-based optimization)

    Attributes:
        stage_id: 4 (primary stage for Steps 4-6 combined)
        stage_name: "Parallel Feature Pipeline"

    Example:
        >>> executor = ParallelFeatureExecutor(
        ...     cfg=config,
        ...     embedder=embedder,
        ...     memory=memory_backend
        ... )
        >>> ctx = await executor.execute(ctx)
        >>> print(f"Features: {len(ctx.dot_plasma.get('threads', []))}")
        >>> print(f"Shards: {len(ctx.shards)}")
    """

    stage_id: int = 4  # Primary stage (4-6 combined)
    stage_name: str = "Parallel Feature Pipeline"

    def __init__(
        self,
        cfg: 'Config',
        embedder: 'MatryoshkaEmbeddings',
        memory: Optional[Any] = None,
        retriever: Optional[Any] = None,
        complexity: Optional[Any] = None,
        provenance: Optional[Any] = None,
        linguistic_gate: Optional[Any] = None,
        guardrails: Optional['SafetyGuardrails'] = None,
        multipass_memory_crawl: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None,
        emit_stage_event: Optional[Callable[[int, str, float], None]] = None,
    ):
        """
        Initialize Parallel Feature Executor.

        Args:
            cfg: Configuration object
            embedder: Base Matryoshka embedder for fallback
            memory: Dynamic memory backend (for multipass crawl)
            retriever: Legacy retriever (fallback)
            complexity: Complexity level for retrieval tuning
            provenance: Provenance trace for logging
            linguistic_gate: Optional linguistic gate for Phase 5
            guardrails: Optional safety guardrails
            multipass_memory_crawl: Async function for multipass crawling
            logger: Optional logger instance
            emit_stage_event: Optional callback for stage monitoring
        """
        super().__init__(logger=logger, emit_stage_event=emit_stage_event)
        self.cfg = cfg
        self.embedder = embedder
        self.memory = memory
        self.retriever = retriever
        self.complexity = complexity
        self.provenance = provenance
        self.linguistic_gate = linguistic_gate
        self.guardrails = guardrails
        self.multipass_memory_crawl = multipass_memory_crawl

    async def execute(self, ctx: 'WeavingContext') -> 'WeavingContext':
        """
        Execute parallel feature pipeline.

        Delegates to Phase 1 pure functions:
        1. execute_steps_4_6_parallel (main parallel execution)
        2. execute_step5_5_warp_compute (post-parallel tensor ops)
        3. execute_step6_5_beta_wave_packing (post-parallel packing)

        Args:
            ctx: WeavingContext with pattern_spec and threads

        Returns:
            WeavingContext with dot_plasma, warp_space, shards, and related fields
        """
        self._log_stage_start()
        start = self._start_timing()

        # Import pure functions
        from HoloLoom.orchestrator.stages.steps_4_6 import (
            execute_steps_4_6_parallel,
            execute_step5_5_warp_compute,
            execute_step6_5_beta_wave_packing,
        )

        # Step 4-6: Main parallel execution
        ctx = await execute_steps_4_6_parallel(
            ctx=ctx,
            cfg=self.cfg,
            embedder=self.embedder,
            memory=self.memory,
            retriever=self.retriever,
            complexity=self.complexity,
            provenance=self.provenance,
            linguistic_gate=self.linguistic_gate,
            guardrails=self.guardrails,
            multipass_memory_crawl=self.multipass_memory_crawl,
            emit_stage_event=self._emit_stage_event,
            log=self.logger,
        )

        # Step 5.5: Post-parallel warp compute
        ctx = await execute_step5_5_warp_compute(
            ctx=ctx,
            log=self.logger,
        )

        # Step 6.5: Post-parallel beta wave packing
        ctx = await execute_step6_5_beta_wave_packing(
            ctx=ctx,
            cfg=self.cfg,
            memory=self.memory,
            log=self.logger,
        )

        duration = self._record_timing(ctx, 'parallel_feature_pipeline', start)
        self._log_stage_complete(duration)

        return ctx

    def update_dependencies(
        self,
        memory: Optional[Any] = None,
        retriever: Optional[Any] = None,
        complexity: Optional[Any] = None,
        provenance: Optional[Any] = None,
    ) -> None:
        """
        Update dynamic dependencies after initialization.

        Some dependencies (memory, retriever, complexity, provenance) may change
        between queries. This method allows updating them without recreating
        the executor.

        Args:
            memory: Updated memory backend
            retriever: Updated retriever
            complexity: Updated complexity level
            provenance: Updated provenance trace
        """
        if memory is not None:
            self.memory = memory
        if retriever is not None:
            self.retriever = retriever
        if complexity is not None:
            self.complexity = complexity
        if provenance is not None:
            self.provenance = provenance
