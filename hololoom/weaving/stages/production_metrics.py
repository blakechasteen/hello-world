"""
Production Metrics Stage
========================
Post-pipeline metrics emission for monitoring.

Emits Prometheus metrics and internal performance tracking.
Side-effect stage — no context writes.
"""

import logging
from typing import Any, ClassVar, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class ProductionMetricsStage:
    """
    Production Metrics Stage (Monitoring).

    Emits metrics to Prometheus and internal tracking systems.
    Side-effect stage — writes to metrics backend, not context.
    """

    reads: ClassVar[frozenset[str]] = frozenset({'stage_timings', 'spacetime'})
    writes: ClassVar[frozenset[str]] = frozenset()  # Side-effect only
    cacheable: ClassVar[bool] = False  # Side-effect stage

    def __init__(
        self,
        metrics_collector: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.metrics_collector = metrics_collector
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Emit pipeline metrics."""
        if self.metrics_collector is None:
            return

        try:
            timings = ctx.stage_timings or {}
            total_ms = sum(timings.values())

            self.metrics_collector.record_query(
                total_duration_ms=total_ms,
                stage_timings=timings,
                tool_used=ctx.spacetime.tool_used if ctx.spacetime else None,
                confidence=ctx.spacetime.confidence if ctx.spacetime else 0.0,
                error_count=len(ctx.errors) if ctx.errors else 0,
            )

            self.logger.info(f"  [M] Metrics emitted: {total_ms:.1f}ms total")

        except Exception as e:
            self.logger.warning(f"  [M] Metrics emission failed: {e}")

    def get_stage_name(self) -> str:
        return "production_metrics"

    def get_required_context_keys(self) -> list[str]:
        return []

    def get_output_context_keys(self) -> list[str]:
        return []
