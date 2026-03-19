"""
Warp Tensioning Stage
=====================
Step 5 of the weaving cycle: Tension threads into continuous manifold via WarpSpace.

This stage takes selected threads and tensions them into the warp space,
creating a continuous tensor manifold for downstream compute operations.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class WarpTensioningStage:
    """
    Warp Tensioning Stage (Step 5: Warp Space).

    Tensions selected threads into a continuous manifold for
    attention computation and spectral feature extraction.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'threads', 'thread_texts', 'thread_ids'})
    writes: ClassVar[frozenset[str]] = frozenset({'warp_operations'})
    cacheable: ClassVar[bool] = True

    def __init__(
        self,
        warp_space: Any,
        logger: logging.Logger | None = None,
    ):
        self.warp_space = warp_space
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Tension threads into warp space, writing warp_operations to ctx."""
        await self.warp_space.tension(ctx.thread_texts, thread_ids=ctx.thread_ids)

        ctx.warp_operations = [
            (datetime.now().isoformat(), "tension", len(ctx.thread_ids))
        ]
        self.logger.info(f"  [5] Warp Space tensioned with {len(ctx.thread_ids)} threads")

    def get_stage_name(self) -> str:
        return "warp_tensioning"

    def get_required_context_keys(self) -> list[str]:
        return ["thread_texts", "thread_ids"]

    def get_output_context_keys(self) -> list[str]:
        return ["warp_operations"]
