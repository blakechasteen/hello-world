"""Complexity assessor gate — enrichment only, never short-circuits.

Accepts pre-bound callables (use functools.partial to bind the orchestrator):
    assess_fn(query) -> ComplexityLevel
    provenance_fn(query, complexity) -> ProvenanceTrace
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TYPE_CHECKING

from hololoom.orchestrator.protocols.stage import GateExecutor

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class ComplexityAssessorExecutor(GateExecutor):
    """Assesses query complexity and creates provenance trace."""

    stage_id: int = 0
    stage_name: str = "Complexity Assessment"
    timing_key: str = "complexity_assessment"

    def __init__(
        self,
        assess_fn: Callable,
        provenance_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.assess_fn = assess_fn
        self.provenance_fn = provenance_fn

    async def evaluate(self, ctx: 'WeavingContext') -> Optional[Any]:
        if ctx.complexity is None:
            ctx.complexity = self.assess_fn(ctx.current_query)
            self.logger.info(
                f"[mythRL] Complexity: {ctx.complexity.name} "
                f"({ctx.complexity.value} steps)"
            )

        if self.provenance_fn and ctx.provenance is None:
            ctx.provenance = self.provenance_fn(ctx.current_query, ctx.complexity)

        return None  # Never short-circuits
