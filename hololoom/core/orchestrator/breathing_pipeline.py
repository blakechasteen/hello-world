"""
BreathingPipeline — wraps ExecutorPipeline with INHALE/EXHALE/REST phases.

Composition, not replacement. When chrono is None, falls through to
the unwrapped pipeline. Breathing is additive.

    INHALE (stages 0-3): MetaPrompt, Pattern, Chrono, Thread
        Gather broadly. Dense features. Low sparsity.
    EXHALE (stages 4-6): ParallelFeature, Convergence, Tool
        Decide. Sparse features. Collapse warp space.
    REST   (stage 7+):   Spacetime + mini-consolidation
        Reflect. Consolidate. Apply mini-decay.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hololoom.core.orchestrator.context import WeavingContext
    from hololoom.core.orchestrator.pipeline import ExecutorPipeline

logger = logging.getLogger(__name__)


class BreathingPipeline:
    """Wraps ExecutorPipeline with breathing rhythm.

    The pipeline is divided into three phase groups. ChronoTrigger's
    breathing state is set on each transition. The WeavingContext gets
    a breathing_phase annotation that downstream executors can read.

    Falls through to plain sequential execution when chrono is None.
    """

    def __init__(
        self,
        pipeline: ExecutorPipeline,
        chrono: Any = None,
        memory_layer: Any = None,
        inhale_stages: int = 4,
        exhale_stages: int = 3,
    ):
        self._pipeline = pipeline
        self._chrono = chrono
        self._memory_layer = memory_layer
        self._inhale_end = inhale_stages
        self._exhale_end = inhale_stages + exhale_stages
        self._breath_count = 0

    async def execute(self, ctx: WeavingContext) -> WeavingContext:
        """Execute pipeline with breathing phases."""
        if self._chrono is None:
            return await self._pipeline.execute(ctx)

        executors = self._pipeline.executors
        start = time.monotonic()

        # INHALE: gather broadly
        ctx.breathing_phase = "inhale"
        if self._chrono:
            self._chrono.current_phase = "inhale"
        for ex in executors[: self._inhale_end]:
            ctx = await ex.execute(ctx)

        # EXHALE: decide
        ctx.breathing_phase = "exhale"
        if self._chrono:
            self._chrono.current_phase = "exhale"
        for ex in executors[self._inhale_end : self._exhale_end]:
            ctx = await ex.execute(ctx)

        # REST: reflect + consolidate
        ctx.breathing_phase = "rest"
        if self._chrono:
            self._chrono.current_phase = "rest"
        for ex in executors[self._exhale_end :]:
            ctx = await ex.execute(ctx)

        # Mini-consolidation during REST
        if self._memory_layer is not None:
            consolidator = getattr(self._memory_layer, "_consolidator", None)
            if consolidator is not None:
                try:
                    await consolidator.run_cycle()
                except Exception as e:
                    logger.debug("REST consolidation skipped: %s", e)

        # Reset phase
        if self._chrono:
            self._chrono.current_phase = None
        ctx.breathing_phase = None
        self._breath_count += 1

        elapsed = time.monotonic() - start
        logger.debug(
            "Breath #%d complete (%.1fms)",
            self._breath_count, elapsed * 1000,
        )
        return ctx

    @property
    def breath_count(self) -> int:
        return self._breath_count

    @property
    def pipeline(self) -> ExecutorPipeline:
        """Access the underlying pipeline."""
        return self._pipeline
