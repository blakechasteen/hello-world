"""
Fast Execution Stage
====================
Lightweight result assembly for LITE recipe.

Simplified version of fabric_weaving that skips warp detensioning,
detailed trace construction, and optional metadata injection.
"""

import logging
from datetime import datetime
from typing import ClassVar, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class FastExecutionStage:
    """
    Fast Execution Stage (LITE Variant of Fabric Weaving).

    Assembles Spacetime with minimal overhead — no warp collapse,
    simplified trace, no awareness/semantic metadata injection.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'query', 'collapse_result', 'tool_result'})
    writes: ClassVar[frozenset[str]] = frozenset({'spacetime', 'trace'})
    cacheable: ClassVar[bool] = False

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Assemble minimal Spacetime from tool result."""
        from hololoom.fabric.spacetime import Spacetime, WeavingTrace, Artifact

        end_time = datetime.now()
        start_time = ctx.start_time or end_time
        duration_ms = (end_time - start_time).total_seconds() * 1000

        collapse_result = ctx.collapse_result
        tool_result = ctx.tool_result or {}

        ctx.trace = WeavingTrace(
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            stage_durations=dict(ctx.stage_timings) if ctx.stage_timings else {},
            motifs_detected=[],
            embedding_scales_used=[],
            spectral_features={},
            threads_activated=[],
            context_shards_count=0,
            retrieval_mode="fast",
            policy_adapter=None,
            tool_selected=collapse_result.tool if collapse_result else None,
            tool_confidence=collapse_result.confidence if collapse_result else 0.0,
            bandit_statistics=None,
            warp_operations=[],
            tensor_field_stats={},
            errors=list(ctx.errors) if ctx.errors else [],
            warnings=list(ctx.warnings) if ctx.warnings else [],
        )

        raw_artifacts = tool_result.get('artifacts', [])
        artifacts = [Artifact.from_dict(a) for a in raw_artifacts]

        ctx.spacetime = Spacetime(
            query_text=ctx.query.text,
            response=tool_result.get('result', 'No response'),
            tool_used=collapse_result.tool if collapse_result else "none",
            confidence=collapse_result.confidence if collapse_result else 0.0,
            artifacts=artifacts,
            trace=ctx.trace,
            metadata={'execution_mode': 'lite'},
            context_summary="0 shards",
            sources_used=[],
        )

        self.logger.info(f"  [FE] Fast execution complete ({duration_ms:.1f}ms)")

    def get_stage_name(self) -> str:
        return "fast_execution"

    def get_required_context_keys(self) -> list[str]:
        return ["collapse_result", "tool_result"]

    def get_output_context_keys(self) -> list[str]:
        return ["spacetime", "trace"]
