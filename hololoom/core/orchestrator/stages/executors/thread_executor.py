#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thread Selection Executor - Step 3: Yarn Graph / Shuttle
=========================================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 2: Stage Executor Classes

Executor class that wraps execute_step3_thread_selection pure function.
Selects relevant memory threads from Yarn Graph or Shuttle.

Author: Claude Code (Elegance Pass - Phase 2)
Date: 2025-12-09
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Any, TYPE_CHECKING

from HoloLoom.core.orchestrator.protocols import BaseStageExecutor

if TYPE_CHECKING:
    from HoloLoom.core.orchestrator.context import WeavingContext
    from HoloLoom.core.memory.graph import KG


class ThreadSelectionExecutor(BaseStageExecutor):
    """
    Step 3: Thread Selection (Yarn Graph / Shuttle) Executor.

    Selects relevant memory threads using either:
    - Shuttle: MCTS-powered Warp<->Yarn intersection (advanced)
    - Yarn Graph: Simple temporal thread selection (fallback)

    Attributes:
        stage_id: 3
        stage_name: "Thread Selection"

    Example:
        >>> executor = ThreadSelectionExecutor(
        ...     yarn_graph=kg,
        ...     enable_shuttle=False
        ... )
        >>> ctx = await executor.execute(ctx)
        >>> print(f"Selected {len(ctx.threads)} threads")
    """

    stage_id: int = 3
    stage_name: str = "Thread Selection"

    def __init__(
        self,
        yarn_graph: 'KG',
        shuttle_stage: Optional[Any] = None,
        enable_shuttle: bool = False,
        logger: Optional[logging.Logger] = None,
        emit_stage_event: Optional[Callable[[int, str, float], None]] = None,
    ):
        """
        Initialize Thread Selection Executor.

        Args:
            yarn_graph: KG instance for thread selection
            shuttle_stage: Optional ShuttleStage for advanced selection
            enable_shuttle: Whether to use Shuttle (if available)
            logger: Optional logger instance
            emit_stage_event: Optional callback for stage monitoring
        """
        super().__init__(logger=logger, emit_stage_event=emit_stage_event)
        self.yarn_graph = yarn_graph
        self.shuttle_stage = shuttle_stage
        self.enable_shuttle = enable_shuttle

    async def execute(self, ctx: 'WeavingContext') -> 'WeavingContext':
        """
        Execute thread selection.

        Delegates to Phase 1 pure function execute_step3_thread_selection.

        Args:
            ctx: WeavingContext with temporal_window

        Returns:
            WeavingContext with threads, thread_ids, and thread_texts populated
        """
        self._log_stage_start()
        start = self._start_timing()

        # Import and delegate to pure function
        from HoloLoom.core.orchestrator.stages.steps_0_3 import execute_step3_thread_selection

        ctx = await execute_step3_thread_selection(
            ctx=ctx,
            yarn_graph=self.yarn_graph,
            shuttle_stage=self.shuttle_stage,
            enable_shuttle=self.enable_shuttle,
            emit_stage_event=self._emit_stage_event,
            log=self.logger,
        )

        duration = self._record_timing(ctx, 'thread_selection', start)
        self._log_stage_complete(duration)

        return ctx
