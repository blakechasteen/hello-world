"""
Thread Selection Stage
======================
Step 3 of the weaving cycle: Select threads from Yarn Graph or Shuttle.

Routes to either MCTS-powered Shuttle (warp-yarn intersection) or
fallback Yarn Graph for thread selection based on temporal window.
"""

import logging
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class ThreadSelectionStage:
    """
    Thread Selection Stage (Step 3: Yarn Graph / Shuttle).

    Selects threads from knowledge graph based on temporal window
    and query relevance.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'query', 'pattern_spec', 'temporal_window'})
    writes: ClassVar[frozenset[str]] = frozenset({'threads', 'thread_ids', 'thread_texts', 'shuttle_used'})
    cacheable: ClassVar[bool] = True

    def __init__(
        self,
        yarn_graph: Any,
        shuttle_stage: Any | None = None,
        enable_shuttle: bool = False,
        logger: logging.Logger | None = None,
    ):
        self.yarn_graph = yarn_graph
        self.shuttle_stage = shuttle_stage
        self.enable_shuttle = enable_shuttle
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Select threads from Shuttle or Yarn Graph, writing to ctx."""
        if self.enable_shuttle and self.shuttle_stage:
            ctx.threads = await self.shuttle_stage.select_threads(
                temporal_window=ctx.temporal_window,
                query=ctx.query,
                trajectory_name=None,
            )
            ctx.shuttle_used = True
            self.logger.info(f"  [3] Shuttle selected {len(ctx.threads)} threads")
        else:
            ctx.threads = self.yarn_graph.select_threads(ctx.temporal_window, ctx.query)
            ctx.shuttle_used = False
            self.logger.info(f"  [3] Yarn Graph selected {len(ctx.threads)} threads")

        ctx.thread_ids = [s.id for s in ctx.threads]
        ctx.thread_texts = [s.text for s in ctx.threads]

    def get_stage_name(self) -> str:
        return "thread_selection"

    def get_required_context_keys(self) -> list[str]:
        return ["temporal_window"]

    def get_output_context_keys(self) -> list[str]:
        return ["threads", "thread_ids", "thread_texts"]
