#!/usr/bin/env python3
"""
Meta-Prompt Executor - Step 0: Query Enhancement
=================================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 2: Stage Executor Classes

Executor class that wraps execute_step0_meta_prompt pure function.
Provides optional query enhancement through LLM calls.

Author: Claude Code (Elegance Pass - Phase 2)
Date: 2025-12-09
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from hololoom.orchestrator.protocols import BaseStageExecutor

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class MetaPromptExecutor(BaseStageExecutor):
    """
    Step 0: Meta-Prompt Enhancement Executor.

    Optionally enhances the query through an LLM call before processing.
    This allows for query rewriting, clarification, or expansion.

    Attributes:
        stage_id: 0 (first stage)
        stage_name: "Meta-Prompt Enhancement"

    Example:
        >>> executor = MetaPromptExecutor(
        ...     enable_enhancement=True,
        ...     proto_llm_call=orchestrator.proto_llm_call
        ... )
        >>> ctx = await executor.execute(ctx)
        >>> if ctx.enhancement_applied:
        ...     print(f"Enhanced: {ctx.current_query_text}")
    """

    stage_id: int = 0
    stage_name: str = "Meta-Prompt Enhancement"

    def __init__(
        self,
        enable_enhancement: bool = False,
        proto_llm_call: Callable[[str], Any] | None = None,
        logger: logging.Logger | None = None,
        emit_stage_event: Callable[[int, str, float], None] | None = None,
    ):
        """
        Initialize Meta-Prompt Executor.

        Args:
            enable_enhancement: Whether to attempt query enhancement
            proto_llm_call: Async callable for LLM query enhancement
            logger: Optional logger instance
            emit_stage_event: Optional callback for stage monitoring
        """
        super().__init__(logger=logger, emit_stage_event=emit_stage_event)
        self.enable_enhancement = enable_enhancement
        self.proto_llm_call = proto_llm_call

    async def execute(self, ctx: WeavingContext) -> WeavingContext:
        """
        Execute meta-prompt enhancement.

        Delegates to Phase 1 pure function execute_step0_meta_prompt.

        Args:
            ctx: WeavingContext with query to potentially enhance

        Returns:
            WeavingContext with enhanced_query set if enhancement applied
        """
        self._log_stage_start()
        start = self._start_timing()

        # Import and delegate to pure function
        from hololoom.orchestrator.stages.steps_0_3 import execute_step0_meta_prompt

        ctx = await execute_step0_meta_prompt(
            ctx=ctx,
            enable_enhancement=self.enable_enhancement,
            proto_llm_call=self.proto_llm_call,
            log=self.logger,
        )

        duration = self._record_timing(ctx, 'meta_prompt_enhancement', start)
        self._log_stage_complete(duration)

        return ctx
