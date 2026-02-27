#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern Selection Executor - Step 1: Loom Command
==================================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 2: Stage Executor Classes

Executor class that wraps execute_step1_pattern_selection pure function.
Selects appropriate pattern card (BARE/FAST/FUSED) based on query.

Author: Claude Code (Elegance Pass - Phase 2)
Date: 2025-12-09
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, TYPE_CHECKING

from hololoom.core.orchestrator.protocols import BaseStageExecutor

if TYPE_CHECKING:
    from hololoom.core.orchestrator.context import WeavingContext
    from hololoom.core.loom.command import LoomCommand


class PatternSelectionExecutor(BaseStageExecutor):
    """
    Step 1: Pattern Selection (Loom Command) Executor.

    Analyzes the query and selects an appropriate processing pattern:
    - BARE: Minimal processing (<50ms)
    - FAST: Balanced processing (<150ms)
    - FUSED: Full processing (<300ms)

    Attributes:
        stage_id: 1
        stage_name: "Loom Command"

    Example:
        >>> executor = PatternSelectionExecutor(loom_command=loom)
        >>> ctx = await executor.execute(ctx)
        >>> print(f"Selected: {ctx.pattern_spec.name}")
    """

    stage_id: int = 1
    stage_name: str = "Loom Command"

    def __init__(
        self,
        loom_command: 'LoomCommand',
        logger: Optional[logging.Logger] = None,
        emit_stage_event: Optional[Callable[[int, str, float], None]] = None,
    ):
        """
        Initialize Pattern Selection Executor.

        Args:
            loom_command: LoomCommand instance for pattern selection
            logger: Optional logger instance
            emit_stage_event: Optional callback for stage monitoring
        """
        super().__init__(logger=logger, emit_stage_event=emit_stage_event)
        self.loom_command = loom_command

    async def execute(self, ctx: 'WeavingContext') -> 'WeavingContext':
        """
        Execute pattern selection.

        Delegates to Phase 1 pure function execute_step1_pattern_selection.

        Args:
            ctx: WeavingContext with query to classify

        Returns:
            WeavingContext with pattern_spec set
        """
        self._log_stage_start()
        start = self._start_timing()

        # Import and delegate to pure function
        from hololoom.core.orchestrator.stages.steps_0_3 import execute_step1_pattern_selection

        ctx = await execute_step1_pattern_selection(
            ctx=ctx,
            loom_command=self.loom_command,
            emit_stage_event=self._emit_stage_event,
            log=self.logger,
        )

        duration = self._record_timing(ctx, 'pattern_selection', start)
        self._log_stage_complete(duration)

        return ctx
