#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spacetime Executor - Step 9: Fabric Assembly
=============================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 2: Stage Executor Classes

Executor class that wraps execute_step9_spacetime_fabric pure function.
Assembles the final Spacetime output with complete provenance.

Author: Claude Code (Elegance Pass - Phase 2)
Date: 2025-12-09
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Any, Dict, TYPE_CHECKING

from HoloLoom.orchestrator.protocols import BaseStageExecutor

if TYPE_CHECKING:
    from HoloLoom.orchestrator.context import WeavingContext
    from HoloLoom.config import Config


class SpacetimeExecutor(BaseStageExecutor):
    """
    Step 9: Spacetime Fabric Assembly Executor.

    Assembles the final Spacetime output with complete provenance:
    - Detensions Warp Space (collapse continuous → discrete)
    - Creates WeavingTrace with full stage timings and metadata
    - Assembles Spacetime artifact with response, confidence, and provenance
    - Optionally injects awareness context and dashboard

    Attributes:
        stage_id: 9
        stage_name: "Spacetime Fabric"

    Example:
        >>> executor = SpacetimeExecutor(
        ...     cfg=config,
        ...     semantic_cache=cache,
        ...     dashboard_constructor=dashboard
        ... )
        >>> ctx = await executor.execute(ctx)
        >>> print(f"Response: {ctx.spacetime.response}")
        >>> print(f"Confidence: {ctx.spacetime.confidence:.2f}")
        >>> print(f"Trace: {ctx.spacetime.trace.duration_ms:.1f}ms")
    """

    stage_id: int = 9
    stage_name: str = "Spacetime Fabric"

    def __init__(
        self,
        cfg: 'Config',
        semantic_cache: Optional[Any] = None,
        dashboard_constructor: Optional[Any] = None,
        awareness_context: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        emit_stage_event: Optional[Callable[[int, str, float], None]] = None,
    ):
        """
        Initialize Spacetime Executor.

        Args:
            cfg: Configuration
            semantic_cache: Optional semantic cache for statistics
            dashboard_constructor: Optional dashboard constructor
            awareness_context: Optional awareness/consciousness context
            logger: Optional logger instance
            emit_stage_event: Optional callback for stage monitoring
        """
        super().__init__(logger=logger, emit_stage_event=emit_stage_event)
        self.cfg = cfg
        self.semantic_cache = semantic_cache
        self.dashboard_constructor = dashboard_constructor
        self.awareness_context = awareness_context

    async def execute(self, ctx: 'WeavingContext') -> 'WeavingContext':
        """
        Execute spacetime fabric assembly.

        Delegates to Phase 1 pure function execute_step9_spacetime_fabric.

        Args:
            ctx: WeavingContext with all previous step results

        Returns:
            WeavingContext with spacetime and trace set
        """
        self._log_stage_start()
        start = self._start_timing()

        # Import and delegate to pure function
        from HoloLoom.orchestrator.stages.steps_7_9 import execute_step9_spacetime_fabric

        ctx = await execute_step9_spacetime_fabric(
            ctx=ctx,
            cfg=self.cfg,
            semantic_cache=self.semantic_cache,
            dashboard_constructor=self.dashboard_constructor,
            awareness_context=self.awareness_context,
            emit_stage_event=self._emit_stage_event,
            log=self.logger,
        )

        duration = self._record_timing(ctx, 'spacetime_assembly', start)
        self._log_stage_complete(duration)

        return ctx

    def update_semantic_cache(self, semantic_cache: Optional[Any]) -> None:
        """
        Update semantic cache.

        Args:
            semantic_cache: New semantic cache (or None to disable)
        """
        self.semantic_cache = semantic_cache

    def update_dashboard_constructor(self, dashboard_constructor: Optional[Any]) -> None:
        """
        Update dashboard constructor.

        Args:
            dashboard_constructor: New dashboard constructor (or None to disable)
        """
        self.dashboard_constructor = dashboard_constructor

    def update_awareness_context(self, awareness_context: Optional[Dict[str, Any]]) -> None:
        """
        Update awareness context for consciousness integration.

        Args:
            awareness_context: New awareness context dict
        """
        self.awareness_context = awareness_context
