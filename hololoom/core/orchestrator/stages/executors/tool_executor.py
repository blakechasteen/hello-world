#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool Execution Executor - Step 8: Safety-Gated Tool Execution
==============================================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 2: Stage Executor Classes

Executor class that wraps execute_step8_tool_execution pure function.
Executes selected tool with safety gating via SafetyGuardrails.

Author: Claude Code (Elegance Pass - Phase 2)
Date: 2025-12-09
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, TYPE_CHECKING

from hololoom.core.orchestrator.protocols import BaseStageExecutor

if TYPE_CHECKING:
    from hololoom.core.orchestrator.context import WeavingContext
    from hololoom.tools.executor import ToolExecutor
    from hololoom.alignment.safety_guardrails import SafetyGuardrails
    from hololoom.alignment.audit_trail import AuditTrail


class ToolExecutionExecutor(BaseStageExecutor):
    """
    Step 8: Tool Execution with Safety Gating Executor.

    Executes the selected tool with comprehensive safety checks:
    - Gates action through SafetyGuardrails (risk-based evaluation)
    - Logs decision to AuditTrail for compliance/debugging
    - Executes tool if allowed, blocks if safety check fails
    - Supports graceful degradation if safety components unavailable

    Attributes:
        stage_id: 8
        stage_name: "Tool Execution"

    Example:
        >>> executor = ToolExecutionExecutor(
        ...     tool_executor=tool_executor,
        ...     guardrails=safety_guardrails,
        ...     audit_trail=audit_trail
        ... )
        >>> ctx = await executor.execute(ctx)
        >>> if ctx.safety_blocked:
        ...     print(f"Blocked: {ctx.safety_decision.reason}")
        >>> else:
        ...     print(f"Result: {ctx.tool_result}")
    """

    stage_id: int = 8
    stage_name: str = "Tool Execution"

    def __init__(
        self,
        tool_executor: 'ToolExecutor',
        guardrails: Optional['SafetyGuardrails'] = None,
        audit_trail: Optional['AuditTrail'] = None,
        logger: Optional[logging.Logger] = None,
        emit_stage_event: Optional[Callable[[int, str, float], None]] = None,
    ):
        """
        Initialize Tool Execution Executor.

        Args:
            tool_executor: Tool executor for running tools
            guardrails: Optional safety guardrails for action gating
            audit_trail: Optional audit trail for logging decisions
            logger: Optional logger instance
            emit_stage_event: Optional callback for stage monitoring
        """
        super().__init__(logger=logger, emit_stage_event=emit_stage_event)
        self.tool_executor = tool_executor
        self.guardrails = guardrails
        self.audit_trail = audit_trail

    async def execute(self, ctx: 'WeavingContext') -> 'WeavingContext':
        """
        Execute tool with safety gating.

        Delegates to Phase 1 pure function execute_step8_tool_execution.

        Args:
            ctx: WeavingContext with collapse_result from Step 7

        Returns:
            WeavingContext with tool_result, safety_decision set.
            If blocked by safety, safety_blocked=True.
        """
        self._log_stage_start()
        start = self._start_timing()

        # Import and delegate to pure function
        from hololoom.core.orchestrator.stages.steps_7_9 import execute_step8_tool_execution

        ctx = await execute_step8_tool_execution(
            ctx=ctx,
            tool_executor=self.tool_executor,
            guardrails=self.guardrails,
            audit_trail=self.audit_trail,
            emit_stage_event=self._emit_stage_event,
            log=self.logger,
        )

        duration = self._record_timing(ctx, 'tool_execution', start)
        self._log_stage_complete(duration)

        return ctx

    def update_guardrails(self, guardrails: Optional['SafetyGuardrails']) -> None:
        """
        Update safety guardrails.

        Args:
            guardrails: New guardrails (or None to disable)
        """
        self.guardrails = guardrails

    def update_audit_trail(self, audit_trail: Optional['AuditTrail']) -> None:
        """
        Update audit trail.

        Args:
            audit_trail: New audit trail (or None to disable)
        """
        self.audit_trail = audit_trail
