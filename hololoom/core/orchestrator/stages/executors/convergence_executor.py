#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convergence Executor - Step 7: Decision Collapse
=================================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 2: Stage Executor Classes

Executor class that wraps execute_step7_convergence pure function.
Collapses probability distributions to discrete tool selections.

Author: Claude Code (Elegance Pass - Phase 2)
Date: 2025-12-09
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Any, TYPE_CHECKING

from hololoom.core.orchestrator.protocols import BaseStageExecutor

if TYPE_CHECKING:
    from hololoom.core.orchestrator.context import WeavingContext
    from hololoom.config import Config
    from hololoom.tools.executor import ToolExecutor


class ConvergenceExecutor(BaseStageExecutor):
    """
    Step 7: Convergence Engine (Decision Collapse) Executor.

    Collapses probability distributions to discrete tool selections:
    - Gets neural predictions from policy engine
    - Optionally blends with gradient flow routing
    - Uses Thompson Sampling for exploration/exploitation balance
    - Selects final tool via collapse strategies (ARGMAX, EPSILON_GREEDY, etc.)

    Attributes:
        stage_id: 7
        stage_name: "Convergence Engine"

    Example:
        >>> executor = ConvergenceExecutor(
        ...     cfg=config,
        ...     policy=policy_engine,
        ...     tool_executor=tool_executor
        ... )
        >>> ctx = await executor.execute(ctx)
        >>> print(f"Selected: {ctx.collapse_result.tool}")
        >>> print(f"Confidence: {ctx.collapse_result.confidence:.2f}")
    """

    stage_id: int = 7
    stage_name: str = "Convergence Engine"

    def __init__(
        self,
        cfg: 'Config',
        policy: Any,
        tool_executor: 'ToolExecutor',
        gradient_router: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        emit_stage_event: Optional[Callable[[int, str, float], None]] = None,
    ):
        """
        Initialize Convergence Executor.

        Args:
            cfg: Configuration (bandit strategy, epsilon)
            policy: Policy engine for neural predictions
            tool_executor: Tool executor (for available tools list)
            gradient_router: Optional gradient flow router
            logger: Optional logger instance
            emit_stage_event: Optional callback for stage monitoring
        """
        super().__init__(logger=logger, emit_stage_event=emit_stage_event)
        self.cfg = cfg
        self.policy = policy
        self.tool_executor = tool_executor
        self.gradient_router = gradient_router

    async def execute(self, ctx: 'WeavingContext') -> 'WeavingContext':
        """
        Execute convergence (decision collapse).

        Delegates to Phase 1 pure function execute_step7_convergence.

        Args:
            ctx: WeavingContext with features and retrieval_context

        Returns:
            WeavingContext with action_plan, collapse_result, neural_probs set
        """
        self._log_stage_start()
        start = self._start_timing()

        # Import and delegate to pure function
        from hololoom.core.orchestrator.stages.steps_7_9 import execute_step7_convergence

        ctx = await execute_step7_convergence(
            ctx=ctx,
            cfg=self.cfg,
            policy=self.policy,
            tool_executor=self.tool_executor,
            gradient_router=self.gradient_router,
            emit_stage_event=self._emit_stage_event,
            log=self.logger,
        )

        duration = self._record_timing(ctx, 'convergence', start)
        self._log_stage_complete(duration)

        return ctx

    def update_policy(self, policy: Any) -> None:
        """
        Update the policy engine.

        Args:
            policy: New policy engine
        """
        self.policy = policy

    def update_gradient_router(self, gradient_router: Optional[Any]) -> None:
        """
        Update the gradient router.

        Args:
            gradient_router: New gradient router (or None to disable)
        """
        self.gradient_router = gradient_router
