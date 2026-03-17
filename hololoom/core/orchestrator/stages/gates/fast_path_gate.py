"""Fast-path routing gate — shortcuts TRIVIAL/SIMPLE queries."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from hololoom.orchestrator.protocols.stage import GateExecutor

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class FastPathGateExecutor(GateExecutor):
    """Routes simple queries to fast-path, skipping the full pipeline."""

    stage_id: int = -2
    stage_name: str = "Fast Path Routing"
    timing_key: str = "fast_path_routing"

    def __init__(self, query_classifier: Any, fast_path_router: Any, **kwargs):
        super().__init__(**kwargs)
        self.query_classifier = query_classifier
        self.fast_path_router = fast_path_router

    async def evaluate(self, ctx: 'WeavingContext') -> Optional[Any]:
        if not self.query_classifier or not self.fast_path_router:
            return None

        classification = self.query_classifier.classify(ctx.current_query_text)
        ctx.query_classification = classification

        from hololoom.routing.query_classifier import QueryComplexity
        if classification.complexity in {QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE}:
            self.logger.info(
                f"[ROUTING] Fast path: {classification.complexity.value} "
                f"(confidence={classification.confidence:.2f}, reason={classification.reasoning})"
            )
            spacetime = await self.fast_path_router.route(ctx.current_query, classification)
            ctx.fast_path_used = True
            self.logger.debug(
                f"[ROUTING] Fast path completed in "
                f"{spacetime.metadata.get('latency_ms', 0):.1f}ms"
            )
            return spacetime

        self.logger.info(
            f"[ROUTING] Full pipeline: {classification.complexity.value} "
            f"(confidence={classification.confidence:.2f}, reason={classification.reasoning})"
        )
        return None
