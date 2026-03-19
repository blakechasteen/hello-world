"""
GovernancePipeline — builds and executes the middleware chain.
"""

from __future__ import annotations

import logging

from .context import GovernanceContext
from .protocol import GovernanceMiddleware, NextFn

logger = logging.getLogger(__name__)


class GovernancePipeline:
    """Ordered chain of GovernanceMiddleware executed in sequence.

    Usage:
        pipeline = GovernancePipeline()
        pipeline.add(RBACMiddleware(governance_engine))
        pipeline.add(SafetyGateMiddleware(safety_guardrails))
        pipeline.add(ReasoningMiddleware(get_orchestrator))
        pipeline.add(AuditTrailMiddleware(audit_trail))
        pipeline.add(DeceptionMonitorMiddleware(deception_detector))

        ctx = GovernanceContext(query="hello")
        ctx = await pipeline.execute(ctx)
    """

    def __init__(self) -> None:
        self._middleware: list[GovernanceMiddleware] = []

    def add(self, mw: GovernanceMiddleware) -> GovernancePipeline:
        """Append middleware to the chain. Returns self for fluent API."""
        self._middleware.append(mw)
        logger.debug(f"Governance middleware added: {mw.name}")
        return self

    async def execute(self, ctx: GovernanceContext) -> GovernanceContext:
        """Build the chain in reverse and execute from the first middleware."""
        if not self._middleware:
            logger.warning("GovernancePipeline.execute() called with no middleware")
            return ctx

        # Terminal handler — identity (returns ctx unchanged)
        async def terminal(c: GovernanceContext) -> GovernanceContext:
            return c

        # Build chain in reverse so first-added middleware runs first
        chain: NextFn = terminal
        for mw in reversed(self._middleware):
            chain = _wrap(mw, chain)

        logger.debug(
            f"Executing governance pipeline "
            f"({len(self._middleware)} middleware, "
            f"correlation={ctx.correlation_id})"
        )
        return await chain(ctx)

    @property
    def middleware_names(self) -> list[str]:
        return [mw.name for mw in self._middleware]

    def __len__(self) -> int:
        return len(self._middleware)


def _wrap(mw: GovernanceMiddleware, next_fn: NextFn) -> NextFn:
    """Wrap a middleware + its successor into a single NextFn."""

    async def wrapped(ctx: GovernanceContext) -> GovernanceContext:
        return await mw(ctx, next_fn)

    return wrapped
