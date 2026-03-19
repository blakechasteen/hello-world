"""
Reflection Stage
================
Post-execution reflection and episodic buffer recording.

Records query-response outcomes for temporal pattern analysis
and quality degradation detection.
Side-effect stage — no context writes.
"""

import logging
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class ReflectionStage:
    """
    Reflection Stage (Episodic Buffer).

    Records execution outcomes to the reflection buffer for
    temporal pattern analysis, trend detection, and quality monitoring.
    Side-effect stage — writes to reflection buffer, not context.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'spacetime', 'collapse_result'})
    writes: ClassVar[frozenset[str]] = frozenset()  # Side-effect only
    cacheable: ClassVar[bool] = False  # Side-effect stage

    def __init__(
        self,
        reflection_buffer: Any | None = None,
        logger: logging.Logger | None = None,
    ):
        self.reflection_buffer = reflection_buffer
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Record execution outcome to reflection buffer."""
        if self.reflection_buffer is None or ctx.spacetime is None:
            return

        try:
            await self.reflection_buffer.record(
                spacetime=ctx.spacetime,
                collapse_result=ctx.collapse_result,
                confidence=ctx.collapse_result.confidence if ctx.collapse_result else 0.0,
            )
            self.logger.info("  [RF] Reflection recorded")

        except Exception as e:
            self.logger.warning(f"  [RF] Reflection recording failed: {e}")

    def get_stage_name(self) -> str:
        return "reflection"

    def get_required_context_keys(self) -> list[str]:
        return ["spacetime"]

    def get_output_context_keys(self) -> list[str]:
        return []
