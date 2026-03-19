"""Cache gate — returns cached Spacetime on hit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hololoom.orchestrator.protocols.stage import GateExecutor

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class CacheGateExecutor(GateExecutor):
    """Checks query cache before running the full pipeline."""

    stage_id: int = -1
    stage_name: str = "Cache Check"
    timing_key: str = "cache_check"

    def __init__(self, cache: Any = None, **kwargs):
        # Filter kwargs to only those accepted by BaseStageExecutor
        super().__init__(
            logger=kwargs.get('logger'),
            emit_stage_event=kwargs.get('emit_stage_event'),
        )
        self.cache = cache

    async def evaluate(self, ctx: WeavingContext) -> Any | None:
        if not self.cache:
            return None

        cached_result = self.cache.get(ctx.current_query_text)
        if cached_result is not None:
            self.logger.info("[CACHE HIT] Returning cached result for query")
            ctx.cache_hit = True
            return cached_result

        return None
