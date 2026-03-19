"""Rate limit gate — hard rejection, not graceful degradation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hololoom.orchestrator.protocols import BaseStageExecutor

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class RateLimitGateExecutor(BaseStageExecutor):
    """Checks rate limit before any processing. Raises on exceed."""

    stage_id: int = -5
    stage_name: str = "Rate Limit"

    def __init__(self, rate_limiter: Any, **kwargs):
        super().__init__(**kwargs)
        self.rate_limiter = rate_limiter

    async def execute(self, ctx: WeavingContext) -> WeavingContext:
        self._log_stage_start()
        start = self._start_timing()

        if self.rate_limiter and not await self.rate_limiter.acquire():
            self.logger.warning("[PRODUCTION] Rate limit exceeded")
            from hololoom.context.rate_limiter import RateLimitExceededError
            raise RateLimitExceededError(
                f"Rate limit exceeded for query: {ctx.current_query_text[:50]}"
            )

        self._record_timing(ctx, 'rate_limit_check', start)
        return ctx
