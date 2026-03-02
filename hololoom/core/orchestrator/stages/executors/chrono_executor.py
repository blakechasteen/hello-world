#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chrono Trigger Executor - Step 2: Temporal Window
==================================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 2: Stage Executor Classes

Executor class that wraps execute_step2_chrono_trigger pure function.
Creates temporal context for memory retrieval.

Author: Claude Code (Elegance Pass - Phase 2)
Date: 2025-12-09
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, TYPE_CHECKING

from hololoom.orchestrator.protocols import BaseStageExecutor

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class ChronoTriggerExecutor(BaseStageExecutor):
    """
    Step 2: Chrono Trigger (Temporal Window) Executor.

    Creates temporal context for memory retrieval:
    - Defines the time window for relevant memories
    - Sets recency bias for temporal weighting
    - Configures pipeline timeout from pattern spec

    Attributes:
        stage_id: 2
        stage_name: "Chrono Trigger"

    Example:
        >>> executor = ChronoTriggerExecutor(lookback_days=365)
        >>> ctx = await executor.execute(ctx)
        >>> print(f"Window: {ctx.temporal_window.start} to {ctx.temporal_window.end}")
    """

    stage_id: int = 2
    stage_name: str = "Chrono Trigger"

    def __init__(
        self,
        lookback_days: int = 365,
        recency_bias: float = 0.5,
        logger: Optional[logging.Logger] = None,
        emit_stage_event: Optional[Callable[[int, str, float], None]] = None,
    ):
        """
        Initialize Chrono Trigger Executor.

        Args:
            lookback_days: How far back to look for memories (default: 365)
            recency_bias: Weight for recent memories 0-1 (default: 0.5)
            logger: Optional logger instance
            emit_stage_event: Optional callback for stage monitoring
        """
        super().__init__(logger=logger, emit_stage_event=emit_stage_event)
        self.lookback_days = lookback_days
        self.recency_bias = recency_bias

    async def execute(self, ctx: 'WeavingContext') -> 'WeavingContext':
        """
        Execute chrono trigger.

        Delegates to Phase 1 pure function execute_step2_chrono_trigger.

        Args:
            ctx: WeavingContext with pattern_spec

        Returns:
            WeavingContext with chrono and temporal_window set
        """
        self._log_stage_start()
        start = self._start_timing()

        # Import and delegate to pure function
        from hololoom.orchestrator.stages.steps_0_3 import execute_step2_chrono_trigger

        ctx = await execute_step2_chrono_trigger(
            ctx=ctx,
            emit_stage_event=self._emit_stage_event,
            log=self.logger,
            lookback_days=self.lookback_days,
            recency_bias=self.recency_bias,
        )

        duration = self._record_timing(ctx, 'chrono_trigger', start)
        self._log_stage_complete(duration)

        return ctx
